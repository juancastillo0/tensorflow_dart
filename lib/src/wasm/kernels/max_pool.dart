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

// import {backend_util, KernelConfig, KernelFunc, MaxPool, MaxPoolAttrs, MaxPoolInputs, Tensor4D, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

late final Function(List) _wasmMaxPool;
// : (
//     xId: number, batchSize: number, inputHeight: number, inputWidth: number,
//     filterHeight: number, filterWidth: number, padTop: number, padRight: number,
//     padBottom: number, padLeft: number, dilationHeight: number,
//     dilationWidth: number, strideHeight: number, strideWidth: number,
//     inputChannels: number, outputChannels: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmMaxPool = backend.wasm.cwrap(MaxPool, null /* void */, [
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
    'number', // dilationHeight
    'number', // dilationWidth
    'number', // strideHeight
    'number', // strideWidth
    'number', // inputChannels
    'number', // outputChannels
    'number', // outId
  ]);
}

TensorInfo maxPool({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']! as Tensor4D;
  final xId = backend.dataIdMap.get(x.dataId)!.id;

  // TF API supports int32 input. CPU and WebGL backend also support int32
  // input. WASM backend doesn't support it because it uses xnnpack which only
  // supports float32.
  //
  // Add the following assert only for the WASM backend instead of at core op
  // level.
  //
  // TODO: add support for int32 input.
  util.assert_(
      x.dtype == 'float32',
      () =>
          "Error in MaxPool: only float32 input is supported. Got ${x.dtype}.");

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
  final dilationHeight = convInfo.dilationHeight;
  final dilationWidth = convInfo.dilationWidth;
  final strideHeight = convInfo.strideHeight;
  final strideWidth = convInfo.strideWidth;
  final inputChannels = convInfo.inChannels;
  final outputChannels = convInfo.outChannels;

  if (convInfo.dataFormat != 'channelsLast') {
    throw Exception("wasm backend does not support dataFormat:'" +
        "${convInfo.dataFormat}'. Please use 'channelsLast'.");
  }

  final out = backend.makeOutput(convInfo.outShape, 'float32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  _wasmMaxPool([
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
    dilationHeight,
    dilationWidth,
    strideHeight,
    strideWidth,
    inputChannels,
    outputChannels,
    outId
  ]);
  return out;
}

final maxPoolConfig = KernelConfigG(
  kernelName: MaxPool,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: maxPool,
);
