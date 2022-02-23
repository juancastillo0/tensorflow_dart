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

// import {KernelConfig, KernelFunc, Softmax, SoftmaxAttrs, SoftmaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';

import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmFunc;
// : (xId: number, outId: number, channels: number, batch: number) =>    void;

void _setup(BackendWasm backend) {
  _wasmFunc = backend.wasm.cwrap(Softmax, null /* void */, [
    'number', // xId
    'number', // outId
    'number', // channels
    'number' // batch
  ]);
}

TensorInfo softmax({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final logits = inputs['logits']!;
  final dim = attrs!['dim'] as int;

  final xId = backend.dataIdMap.get(logits.dataId)!.id;
  final out = backend.makeOutput(logits.shape, logits.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final channels = logits.shape[dim];
  final batch = util.sizeFromShape(logits.shape) / channels;

  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(out.shape) == 0) {
    return out;
  }

  _wasmFunc([xId, outId, channels, batch]);
  return out;
}

final softmaxConfig = KernelConfigG(
  kernelName: Softmax,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: softmax,
);
