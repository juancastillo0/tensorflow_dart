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

// import {KernelConfig, KernelFunc, LeakyRelu, LeakyReluAttrs, LeakyReluInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import '_prelude.dart';
import '../../util_base.dart' as util;

late final Function(List) _wasmFunc;
// : (
//     xId: number, dtype: number, leakyreluAlpha: number, outId: number) => void;

void _setupFunc(BackendWasm backend) {
  _wasmFunc = backend.wasm.cwrap(LeakyRelu, null /* void */, [
    'number', // x_id
    'number', // dtype
    'number', // leakyrelu_alpha
    'number', // out_id
  ]);
}

TensorInfo leakyRelu({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final alpha = attrs!['alpha'] as double;

  final xId = backend.dataIdMap.get(x.dataId)!.id;
  // According to TF API, LeakyRelu returns float32 when input is either float32
  // or int32.
  final out = backend.makeOutput(x.shape, 'float32');

  if (util.sizeFromShape(x.shape) != 0) {
    final outId = backend.dataIdMap.get(out.dataId)!.id;
    _wasmFunc([xId, CppDType.values.byName(x.dtype).index, alpha, outId]);
  }

  return out;
}

final leakyReluConfig = KernelConfigG(
  kernelName: LeakyRelu,
  backendName: 'wasm',
  setupFunc: _setupFunc,
  kernelFunc: leakyRelu,
);
