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

// import {AddN, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import 'dart:typed_data';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmFunc;
// (inputIds: Uint8Array, inputIdsLen: number, dtype: number, outId: number) =>
//     void;

void _setupFunc(BackendWasm backend) {
  _wasmFunc = backend.wasm.cwrap(AddN, null /* void */, [
    'array', // input_ids
    'number', // input_ids.length
    'number', // dtype
    'number', // out_id
  ]);
}

TensorInfo addn({
  required List<TensorInfo> inputs,
  required BackendWasm backend,
}) {
  final out = backend.makeOutput(inputs[0].shape, inputs[0].dtype);

  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(out.shape) == 0) {
    return out;
  }

  final inputIds =
      inputs.map((x) => backend.dataIdMap.get(x.dataId)!.id).toList();
  final inputIdsBytes = Uint8List.view(Int32List.fromList(inputIds).buffer);
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  _wasmFunc([
    inputIdsBytes,
    inputIds.length,
    CppDType.values.byName(out.dtype).index,
    outId,
  ]);

  return out;
}

final addNConfig = KernelConfigG(
  kernelName: AddN,
  backendName: 'wasm',
  setupFunc: _setupFunc,
  kernelFunc: ({
    required inputs,
    required backend,
    attrs,
  }) =>
      addn(
    backend: backend,
    inputs: List.generate(inputs.length, (index) => inputs['$index']!),
  ),
);
