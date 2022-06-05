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

// import {KernelConfig, KernelFunc, Sigmoid, SigmoidInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';

import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmFunc; //: (xId: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmFunc =
      backend.wasm.cwrap(Sigmoid, null /* void */, ['number', 'number']);
}

TensorInfo sigmoid({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final out = backend.makeOutput(x.shape, x.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(out.shape) == 0) {
    return out;
  }

  _wasmFunc([xId, outId]);
  return out;
}

final sigmoidConfig = KernelConfigG(
  kernelName: Sigmoid,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: sigmoid,
);
