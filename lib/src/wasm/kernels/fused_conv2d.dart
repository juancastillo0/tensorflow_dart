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

// import {backend_util, FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs, KernelConfig, KernelFunc, Tensor4D} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {FusableActivation} from './types';

import '_prelude.dart';

import 'package:tensorflow_wasm/src/ops/fused_types.dart';
import 'package:collection/collection.dart' hide ListExtensions;

import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

late final Function(List) _wasmFusedConv2d;
// (xId: number, batchSize: number, inputHeight: number, inputWidth: number,
//  filterId: number, filterHeight: number, filterWidth: number,
//  biasId: number, padTop: number, padRight: number, padBottom: number,
//  padLeft: number, isSamePad: number, dilationHeight: number,
//  dilationWidth: number, strideHeight: number, strideWidth: number,
//  inputChannels: number, outputChannels: number, activation: number,
//  preluActivationWeightsId: number, leakyreluAlpha: number, outId: number) =>
//     void;

void _setup(BackendWasm backend) {
  _wasmFusedConv2d = backend.wasm.cwrap(FusedConv2D, null /* void */, [
    'number', // xId
    'number', // batchSize
    'number', // inputHeight
    'number', // inputWidth
    'number', // filterId
    'number', // filterHeight
    'number', // filterWidth
    'number', // biasId
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
    'number', // activation
    'number', // preluActivationWeightsId
    'number', // leakyreluAlpha
    'number', // outId
  ]);
}

TensorInfo fusedConv2d({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
  Function(List)? depthwiseWasmFunc,
}) {
  final depthwise = depthwiseWasmFunc != null;
  final Function(List) wasmFunc = depthwiseWasmFunc ?? _wasmFusedConv2d;
  final opName = depthwise ? 'FusedDepthwiseConv2D' : 'FusedConv2D';

  final x = inputs['x']!;
  final filter = inputs['filter']!;
  final bias = inputs['bias'];
  final preluActivationWeights = inputs['preluActivationWeights'];

  final strides = attrs!['strides'] as List<int>;
  final pad = attrs['pad']!;
  final dilations = attrs['dilations'] as List<int>;
  final dataFormat = attrs['dataFormat'] as String;
  final dimRoundingMode = attrs['dimRoundingMode'] as String?;
  final activation = attrs['activation'] as Activation;
  final leakyreluAlpha = attrs['leakyreluAlpha'] as num?;

  final convInfo = backend_util.computeConv2DInfo(
      (x as Tensor4D).shape, (filter as Tensor4D).shape, strides, dilations,
      pad: pad, roundingMode: dimRoundingMode, depthwise: depthwise);

  final fusedActivation = FusableActivation.values
      .firstWhereOrNull((a) => a.name == activation.name);
  if (fusedActivation == null) {
    throw Exception(
        "${activation} activation not yet supported for ${opName} " +
            "in the wasm backend.");
  }

  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final filterId = backend.dataIdMap.get(filter.dataId)!.id;

  final outputChannels = convInfo.outChannels;

  int biasId = 0;
  if (bias != null) {
    final biasData = backend.dataIdMap.get(bias.dataId)!;
    if (biasData.shape.length != 1) {
      throw Exception("${opName} only supports rank-1 bias but got " +
          "rank ${biasData.shape.length}.");
    }
    if (biasData.shape[0] != outputChannels) {
      throw Exception("${opName} bias shape (${biasData.shape}) does not " +
          "match the number of output channels (${outputChannels})");
    }
    biasId = biasData.id;
  }

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
  final isSamePad = convInfo.padInfo.type == backend_util.PadType.SAME ? 1 : 0;
  final batchSize = convInfo.batchSize;
  final inHeight = convInfo.inHeight;
  final inWidth = convInfo.inWidth;

  if (dataFormat != 'NHWC') {
    throw Exception("wasm backend ${opName} does not support dataFormat:'" +
        "${dataFormat}'. Please use 'NHWC'.");
  }

  final out = backend.makeOutput(convInfo.outShape, 'float32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  final preluActivationWeightsId = preluActivationWeights == null
      ? 0
      : backend.dataIdMap.get(preluActivationWeights.dataId)!.id;

  _wasmFusedConv2d([
    xId,
    batchSize,
    inHeight,
    inWidth,
    filterId,
    filterHeight,
    filterWidth,
    biasId,
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
    fusedActivation.index,
    preluActivationWeightsId,
    leakyreluAlpha ?? 0,
    outId
  ]);

  return out;
}

final fusedConv2DConfig = KernelConfigG(
  kernelName: FusedConv2D,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: fusedConv2d,
);
