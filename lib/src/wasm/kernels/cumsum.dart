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

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;
import 'transpose.dart';

// import {backend_util, KernelConfig, KernelFunc, Cumsum, CumsumAttrs, CumsumInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

// import {transpose} from './Transpose';

late final Function(List) _wasmCumsum;
// : (xId: number, exclusive: number, reverse: number,
//  finalDim: number, outId: number, dtype: CppDType) => void;

void _setup(BackendWasm backend) {
  _wasmCumsum = backend.wasm.cwrap(Cumsum, null /* void */, [
    'number', // x_id
    'number', // exclusive
    'number', // reverse
    'number', // final_dim
    'number', // out_id
    'number' // dtype
  ]);
}

ListOrVal<TensorInfo> cumsum({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final axis = attrs!['axis'] as int;
  final exclusive = attrs['exclusive'] as bool;
  final reverse = attrs['reverse'] as bool;

  final xRank = x.shape.length;

  util.assert_(x.dtype == 'float32' || x.dtype == 'int32',
      () => 'cumsum does not support ${x.dtype} tensors in the WASM backend');
  // permute required axis to inner most axis
  final permutation = backend_util.getAxesPermutation([axis], xRank);
  var permutedX = x;
  if (permutation != null) {
    permutedX = transpose(
      inputs: {'x': x},
      attrs: {'perm': permutation},
      backend: backend,
    ).asVal!;
  }
  final permutedAxis = backend_util.getInnerMostAxes(1, xRank)[0];
  backend_util.assertAxesAreInnerMostDims('cumsum', [permutedAxis], xRank);

  final permutedOut = backend.makeOutput(permutedX.shape, permutedX.dtype);
  final finalDim = permutedX.shape[permutedAxis];
  final permutedXId = backend.dataIdMap.get(permutedX.dataId)!.id;
  final permutedOutId = backend.dataIdMap.get(permutedOut.dataId)!.id;
  _wasmCumsum([
    permutedXId,
    exclusive ? 1 : 0,
    reverse ? 1 : 0,
    finalDim,
    permutedOutId,
    CppDType.values.byName(x.dtype).index,
  ]);

  // transpose data back if permuted
  var out = ListOrVal.val(permutedOut);
  if (permutation != null) {
    final undoPermutation = backend_util.getUndoAxesPermutation(permutation);
    out = transpose(
      inputs: {'x': permutedOut},
      attrs: {'perm': undoPermutation},
      backend: backend,
    );
    backend.disposeData(permutedX.dataId);
    backend.disposeData(permutedOut.dataId);
  }
  return out;
}

final cumsumConfig = KernelConfigG(
  kernelName: Cumsum,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: cumsum,
);
