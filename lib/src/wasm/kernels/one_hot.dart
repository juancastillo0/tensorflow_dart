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

// import {KernelConfig, KernelFunc, OneHot, OneHotAttrs, OneHotInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';

late final Function(List) _wasmOneHot;
// : ( indicesId: number, depth: number, onValue: number, offValue: number,
//     outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmOneHot = backend.wasm.cwrap(OneHot, null /* void */, [
    'number', // indices_id
    'number', // depth,
    'number', // onValue
    'number', // offValue
    'number' // out_id
  ]);
}

TensorInfo oneHot({
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
  required BackendWasm backend,
}) {
  final indices = inputs['indices']!;
  final depth = attrs!['depth'] as int;
  final onValue = attrs['onValue'] as double;
  final offValue = attrs['offValue'] as double;

  final out = backend.makeOutput([...indices.shape, depth], 'int32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final indicesData = backend.dataIdMap.get(indices.dataId)!;
  final indicesId = indicesData.id;

  _wasmOneHot([indicesId, depth, onValue, offValue, outId]);

  return out;
}

final oneHotConfig = KernelConfigG(
  kernelName: OneHot,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: oneHot,
);
