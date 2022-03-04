/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// import {backend_util, Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
import '_prelude.dart';
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;
import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmConv2DBackpropInput;
// : (
//     dyId: number, filterId: number, batchSize: number, filterHeight: number,
//     filterWidth: number, inHeight: number, inWidth: number, inChannels: number,
//     outHeight: number, outWidth: number, outChannels: number,
//     strideHeight: number, strideWidth: number, topPad: number, leftPad: number,
//     fltS0: number, fltS1: number, fltS2: number, xBatchStride: number,
//     xRowStride: number, xColStride: number, xChannelStride: number,
//     yBatchStride: number, yRowStride: number, yColStride: number,
//     yChannelStride: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmConv2DBackpropInput = backend.wasm.cwrap(Conv2DBackpropInput, null, [
    'number', // dyId
    'number', // filterId
    'number', // batchSize
    'number', // filterHeight
    'number', // filterWidth
    'number', // inHeight
    'number', // inWidth
    'number', // inChannels
    'number', // outHeight
    'number', // outWidth
    'number', // outChannels
    'number', // strideHeight
    'number', // strideWidth
    'number', // topPad
    'number', // leftPad
    'number', // fltS0
    'number', // fltS1
    'number', // fltS2
    'number', // xBatchStride
    'number', // xRowStride
    'number', // xColStride
    'number', // xChannelStride
    'number', // yBatchStride
    'number', // yRowStride
    'number', // yColStride
    'number', // yChannelStride
    'number', // outId
  ]);
}

TensorInfo conv2DBackpropInput({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final dy = inputs['dy']!;
  final filter = inputs['filter']!;

  final strides = attrs!['strides'] as List<int>;
  final inputShape = attrs['inputShape'] as List<int>;
  final pad = attrs['pad']!;
  final dimRoundingMode = attrs['dimRoundingMode'] as String?;
  final dataFormat = attrs['dataFormat'] as String;

  final dilations = 1;

  final $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  final convInfo = backend_util.computeConv2DInfo(
      inputShape, filter.shape, strides, [dilations, dilations],
      pad: pad,
      roundingMode: dimRoundingMode,
      depthwise: false /* depthwise */,
      dataFormat: $dataFormat);

  final batchSize = convInfo.batchSize;
  final filterHeight = convInfo.filterHeight;
  final filterWidth = convInfo.filterWidth;
  final inChannels = convInfo.inChannels;
  final inHeight = convInfo.inHeight;
  final inWidth = convInfo.inWidth;
  final outChannels = convInfo.outChannels;
  final outHeight = convInfo.outHeight;
  final outWidth = convInfo.outWidth;
  final strideHeight = convInfo.strideHeight;
  final strideWidth = convInfo.strideWidth;

  final topPad = filterHeight - 1 - convInfo.padInfo.top;
  final leftPad = filterWidth - 1 - convInfo.padInfo.left;

  final isChannelsLast = convInfo.dataFormat == 'channelsLast';
  final dxStrides = util.computeStrides(convInfo.inShape);
  final dyStrides = util.computeStrides(dy.shape);
  final fltStrides = util.computeStrides(filter.shape);
  final xBatchStride = dxStrides[0];
  final xRowStride = isChannelsLast ? dxStrides[1] : dxStrides[2];
  final xColStride = isChannelsLast ? dxStrides[2] : 1;
  final xChannelStride = isChannelsLast ? 1 : dxStrides[1];
  final yBatchStride = dyStrides[0];
  final yRowStride = isChannelsLast ? dyStrides[1] : dyStrides[2];
  final yColStride = isChannelsLast ? dyStrides[2] : 1;
  final yChannelStride = isChannelsLast ? 1 : dyStrides[1];

  final out = backend.makeOutput(convInfo.inShape, 'float32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  final dyId = backend.dataIdMap.get(dy.dataId)!.id;
  final filterId = backend.dataIdMap.get(filter.dataId)!.id;

  _wasmConv2DBackpropInput([
    dyId,
    filterId,
    batchSize,
    filterHeight,
    filterWidth,
    inHeight,
    inWidth,
    inChannels,
    outHeight,
    outWidth,
    outChannels,
    strideHeight,
    strideWidth,
    topPad,
    leftPad,
    fltStrides[0],
    fltStrides[1],
    fltStrides[2],
    xBatchStride,
    xRowStride,
    xColStride,
    xChannelStride,
    yBatchStride,
    yRowStride,
    yColStride,
    yChannelStride,
    outId
  ]);
  return out;
}

final conv2DBackpropInputConfig = KernelConfigG(
  kernelName: Conv2DBackpropInput,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: conv2DBackpropInput,
);
