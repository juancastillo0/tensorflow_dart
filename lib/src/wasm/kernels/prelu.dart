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

// import {KernelConfig, KernelFunc, Prelu, PreluInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {cast} from './Cast';

import '_prelude.dart';
import 'cast.dart';

late final Function(List) _wasmPrelu;
//: (xId: number, weightsId: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmPrelu = backend.wasm.cwrap(Prelu, null /* void */, [
    'number', // x_id
    'number', // weights_id
    'number' // out_id
  ]);
}

TensorInfo prelu({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final alpha = inputs['alpha']!;

  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final weightsId = backend.dataIdMap.get(alpha.dataId)!.id;

  var inputId = xId;
  final input = x;
  var castedInput = input;
  if (input.dtype != 'float32') {
    castedInput =
        cast(backend: backend, inputs: {'x': x}, attrs: {'dtype': 'float32'});
    inputId = backend.dataIdMap.get(castedInput.dataId)!.id;
  }

  final out = backend.makeOutput(x.shape, 'float32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  _wasmPrelu([inputId, weightsId, outId]);

  if (input.dtype != 'float32') {
    backend.disposeData(castedInput.dataId);
  }
  return out;
}

final preluConfig = KernelConfigG(
  kernelName: Prelu,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: prelu,
);
