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

// import {backend_util, FusedDepthwiseConv2D, FusedDepthwiseConv2DAttrs, FusedDepthwiseConv2DInputs, KernelConfig, KernelFunc, Tensor4D} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {FusableActivation} from './types';

import 'package:tensorflow_wasm/src/wasm/kernels/fused_conv2d.dart';

import '_prelude.dart';

late final Function(List) _wasmFusedDepthwiseConv2d;
// (xId: number, batchSize: number, inputHeight: number, inputWidth: number,
//  filterId: number, filterHeight: number, filterWidth: number,
//  biasId: number, padTop: number, padRight: number, padBottom: number,
//  padLeft: number, isSamePad: number, dilationHeight: number,
//  dilationWidth: number, strideHeight: number, strideWidth: number,
//  inputChannels: number, outputChannels: number, activation: number,
//  preluActivationWeightsId: number, leakyreluAlpha: number, outId: number) =>
//     void;

void _setup(BackendWasm backend) {
  _wasmFusedDepthwiseConv2d =
      backend.wasm.cwrap(FusedDepthwiseConv2D, null /* void */, [
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

TensorInfo fusedDepthwiseConv2d({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  return fusedConv2d(
    inputs: inputs,
    backend: backend,
    attrs: attrs,
    depthwiseWasmFunc: _wasmFusedDepthwiseConv2d,
  );
}

final fusedDepthwiseConv2DConfig = KernelConfigG(
  kernelName: FusedDepthwiseConv2D,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: fusedDepthwiseConv2d,
);
