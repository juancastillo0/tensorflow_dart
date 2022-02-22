import 'package:tensorflow_wasm/src/backend_wasm.dart';
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;
import 'package:tensorflow_wasm/src/kernel_names.dart';
import 'package:tensorflow_wasm/src/tensor.dart';

import 'unary_kernel.dart';

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

// import {backend_util, DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs, DepthwiseConv2dNativeInputs, KernelConfig, KernelFunc, Tensor4D} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

late Function(List) _wasmDepthwiseConv2d;
// (
//     xId: number, batchSize: number, inputHeight: number, inputWidth: number,
//     filterId: number, filterHeight: number, filterWidth: number, padTop: number,
//     padRight: number, padBottom: number, padLeft: number, isSamePad: number,
//     dilationHeight: number, dilationWidth: number, strideHeight: number,
//     strideWidth: number, inputChannels: number, outputChannels: number,
//     outId: number) => void;

_setup(BackendWasm backend) {
  _wasmDepthwiseConv2d =
      backend.wasm.cwrap(DepthwiseConv2dNative, null /* void */, [
    'number', // xId
    'number', // batchSize
    'number', // inputHeight
    'number', // inputWidth
    'number', // filterId
    'number', // filterHeight
    'number', // filterWidth
    'number', // padTop
    'number', // padRight
    'number', // padBottom
    'number', // padLeft
    'number', // isSamePad
    'number', // dilationHeight
    'number', // dilationWidth
    'number', // strideHeight
    'number', // strideWidth
    'number', // inputChannels
    'number', // outputChannels
    'number', // outId
  ]);
}

TensorInfo _depthwiseConv2d({
  required DepthwiseConv2dNativeInputs inputs,
  required BackendWasm backend,
  DepthwiseConv2dNativeAttrs? attrs,
}) {
  final x = inputs['x']!;
  final filter = inputs['filter']!;
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final filterId = backend.dataIdMap.get(filter.dataId)!.id;

  final strides = attrs!.strides;
  final dilations = attrs.dilations;
  final pad = attrs.pad;
  final dimRoundingMode = attrs.dimRoundingMode;

  final $dilations = dilations == null ? [1, 1] : dilations;

  final convInfo = backend_util.computeConv2DInfo(
      (x as Tensor4D).shape,
      (filter as Tensor4D).shape,
      strides,
      ($dilations // as number | [number, number]
      ),
      pad: pad,
      roundingMode: dimRoundingMode,
      depthwise: true /* depthwise */);

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
  final isSamePad = convInfo.padInfo.type == backend_util.PadType.SAME ? 1 : 0;

  if (convInfo.dataFormat != 'channelsLast') {
    throw Exception(
        "wasm backend DepthwiseConv2dNative does not support dataFormat:'" +
            "${convInfo.dataFormat}'. Please use 'channelsLast'.");
  }

  final out = backend.makeOutput(convInfo.outShape, 'float32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  _wasmDepthwiseConv2d([
    xId,
    x.shape[0],
    x.shape[1],
    x.shape[2],
    filterId,
    filterHeight,
    filterWidth,
    padTop,
    padRight,
    padBottom,
    padLeft,
    isSamePad,
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

final depthwiseConv2dNativeConfig =
    KernelConfigG<BackendWasm, DepthwiseConv2dNativeAttrs>(
  kernelName: DepthwiseConv2dNative,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: _depthwiseConv2d,
);
