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

// import {KernelConfig, KernelFunc, Step, StepAttrs, StepInputs, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import '_prelude.dart';

late final Function(List) _wasmStep;
// : (xId: number, alpha: number, dtype: number, outId: number) =>
//     void;

void _setup(BackendWasm backend) {
  _wasmStep = backend.wasm.cwrap(Step, null /*void*/, [
    'number', // x_id
    'number', // alpha
    'number', // dtype
    'number', // out_id
  ]);
}

TensorInfo step({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final alpha = attrs!['alpha'] as double;

  final xId = backend.dataIdMap.get(x.dataId)!.id;

  final out = backend.makeOutput(x.shape, x.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  _wasmStep([xId, alpha, CppDType.values.byName(x.dtype).index, outId]);
  return out;
}

final stepConfig = KernelConfigG(
  kernelName: Step,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: step,
);
