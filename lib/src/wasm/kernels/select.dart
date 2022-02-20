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

// import {KernelConfig, KernelFunc, Select, SelectInputs, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmSelect;
// : (    conditionId: number, tId: number, eId: number, offset: number,
//     outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmSelect = backend.wasm.cwrap('SelectV2', null, [
    'number', // conditionId
    'number', // tId
    'number', // eId
    'number', // offset
    'number', // outId
  ]);
}

ListOrVal<TensorInfo> select({
  // SelectInputs
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final condition = inputs['condition']!;
  final t = inputs['t']!;
  final e = inputs['e']!;

  final conditionId = backend.dataIdMap.get(condition.dataId)!.id;
  final tId = backend.dataIdMap.get(t.dataId)!.id;
  final eId = backend.dataIdMap.get(e.dataId)!.id;
  final out = backend.makeOutput(t.shape, t.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final cRank = condition.shape.length;
  final tRank = t.shape.length;

  final offset = cRank == 0 || cRank > 1 || tRank == 1
      ? 1
      : util.sizeFromShape(t.shape.sublist(1));

  _wasmSelect([conditionId, tId, eId, offset, outId]);
  return ListOrVal.val(out);
}

final selectConfig = KernelConfigG(
  kernelName: Select,
  backendName: 'wasm',
  kernelFunc: select,
  setupFunc: _setup,
);
