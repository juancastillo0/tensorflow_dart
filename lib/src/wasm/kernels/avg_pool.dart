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

import '_prelude.dart';
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

// import {AvgPool, AvgPoolAttrs, AvgPoolInputs, backend_util, KernelConfig, KernelFunc, Tensor4D} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

late final Function(List) _wasmAvgPool;
// : (
//     xId: number, batchSize: number, inputHeight: number, inputWidth: number,
//     filterHeight: number, filterWidth: number, padTop: number, padRight: number,
//     padBottom: number, padLeft: number, strideHeight: number,
//     strideWidth: number, channels: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmAvgPool = backend.wasm.cwrap(AvgPool, null /* void */, [
    'number', // xId
    'number', // batchSize
    'number', // inputHeight
    'number', // inputWidth
    'number', // filterHeight
    'number', // filterWidth
    'number', // padTop
    'number', // padRight
    'number', // padBottom
    'number', // padLeft
    'number', // strideHeight
    'number', // strideWidth
    'number', // channels
    'number', // outId
  ]);
}

TensorInfo avgPool({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']! as Tensor4D;
  final xId = backend.dataIdMap.get(x.dataId)!.id;

  final filterSize = attrs!['filterSize'] as List<int>;
  final strides = attrs['strides'] as List<int>;
  final pad = attrs['pad'] as Object;
  final dimRoundingMode = attrs['dimRoundingMode'] as String?;

  final convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides,
      [1, 1] /* dilations */, pad, dimRoundingMode);

  final filterHeight = convInfo.filterHeight;
  final filterWidth = convInfo.filterWidth;
  final padTop = convInfo.padInfo.top;
  final padRight = convInfo.padInfo.right;
  final padBottom = convInfo.padInfo.bottom;
  final padLeft = convInfo.padInfo.left;
  final strideHeight = convInfo.strideHeight;
  final strideWidth = convInfo.strideWidth;
  final channels = convInfo.inChannels;

  if (convInfo.dataFormat != 'channelsLast') {
    throw Exception("wasm backend does not support dataFormat:'" +
        "${convInfo.dataFormat}'. Please use 'channelsLast'.");
  }

  if (convInfo.dilationWidth != 1 || convInfo.dilationHeight != 1) {
    throw Exception(
        "was backend only supports average pooling with dilation = [1, 1], " +
            "got [${convInfo.dilationHeight}, ${convInfo.dilationWidth}].");
  }

  final out = backend.makeOutput(convInfo.outShape, 'float32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  _wasmAvgPool([
    xId,
    x.shape[0],
    x.shape[1],
    x.shape[2],
    filterHeight,
    filterWidth,
    padTop,
    padRight,
    padBottom,
    padLeft,
    strideHeight,
    strideWidth,
    channels,
    outId
  ]);
  return out;
}

final avgPoolConfig = KernelConfigG(
  kernelName: AvgPool,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: avgPool,
);
