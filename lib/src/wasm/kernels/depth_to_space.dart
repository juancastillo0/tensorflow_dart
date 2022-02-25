/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import {DepthToSpace, DepthToSpaceAttrs, DepthToSpaceInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import '_prelude.dart';

late final Function(List) _wasmDepthToSpace;
// : (xId: number, blockSize: number, channelsLast: number, xStrides: Uint8Array,
//     xStridesLength: number, outputShape: Uint8Array, outputStrides: Uint8Array,
//     outSize: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmDepthToSpace = backend.wasm.cwrap(DepthToSpace, null /*void*/, [
    'number', // xId
    'number', // blockSize
    'number', // channelsLast
    'array', // xStrides
    'number', // xStridesLength
    'array', // outputShape
    'array', // outputStrides
    'number', // outSize
    'number', // outId
  ]);
}

TensorInfo depthToSpace({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;

  final blockSize = attrs!['blockSize'] as int;
  final dataFormat = attrs['dataFormat'] as String;

  final batchSize = x.shape[0];
  final inputHeight = (dataFormat == 'NHWC') ? x.shape[1] : x.shape[2];
  final inputWidth = (dataFormat == 'NHWC') ? x.shape[2] : x.shape[3];
  final inputDepth = (dataFormat == 'NHWC') ? x.shape[3] : x.shape[1];

  final outputHeight = inputHeight * blockSize;
  final outputWidth = inputWidth * blockSize;
  final outputDepth = inputDepth ~/ (blockSize * blockSize);

  final outputShape = (dataFormat == 'NHWC')
      ? [batchSize, outputHeight, outputWidth, outputDepth]
      : [batchSize, outputDepth, outputHeight, outputWidth];

  final out = backend.makeOutput(outputShape, 'float32');

  final xData = backend.dataIdMap.get(x.dataId)!;
  final xId = xData.id;
  final xStridesBytes =
      Uint8List.view(Int32List.fromList(util.computeStrides(x.shape)).buffer);

  final outputShapeBytes =
      Uint8List.view(Int32List.fromList(outputShape).buffer);
  final outStridesBytes = Uint8List.view(
      Int32List.fromList(util.computeStrides(outputShape)).buffer);

  final outId = backend.dataIdMap.get(out.dataId)!.id;
  final channelsLast = dataFormat == 'NHWC' ? 1 : 0;
  _wasmDepthToSpace([
    xId,
    blockSize,
    channelsLast,
    xStridesBytes,
    x.shape.length - 1,
    outputShapeBytes,
    outStridesBytes,
    outputShape.length,
    outId
  ]);

  return out;
}

final depthToSpaceConfig = KernelConfigG(
  kernelName: DepthToSpace,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: depthToSpace,
);
