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

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' show SliceList;

import 'kernel_utils.dart';

late final Function(List) _wasmFunc;
// : (xId: number, dtype: number, outerSize: number, innerSize: number,
//     outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmFunc = backend.wasm.cwrap(ArgMax, null /* void */, [
    'number', // x_id
    'number', // dtype
    'number', // outer_size
    'number', // inner_size
    'number' // out_id
  ]);
}

ListOrVal<TensorInfo> argmax({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final axis = attrs!['axis']! as int;
  final x = inputs['x']!;
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  var inputId = xId;
  var input = x;

  final _p = permuteAxesAndTranspose(x, [axis], backend);
  final transposed = _p.transposed;
  final axes = _p.axes;
  final inputWasTransposed = _p.inputWasTransposed;

  if (inputWasTransposed) {
    final transposedId = backend.dataIdMap.get(transposed!.dataId)!.id;
    if (transposedId != xId) {
      // transpose was not a no-op. We will need to dispose of this
      // once we are done.
      input = transposed;
      inputId = transposedId;
    }
  }

  final outShape = input.shape.slice(0, -1);
  final out = backend.makeOutput(outShape, 'int32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  final outerSize = util.sizeFromShape(out.shape);
  final innerSize = input.shape[axes[0]];
  _wasmFunc([
    inputId,
    CppDType.values.byName(input.dtype).index,
    outerSize,
    innerSize,
    outId,
  ]);

  if (inputWasTransposed) {
    // dispose of the transposed tensor.
    backend.disposeData(transposed!.dataId);
  }

  return ListOrVal.val(out);
}

final argMaxConfig = KernelConfigG(
  kernelName: ArgMax,
  backendName: 'wasm',
  kernelFunc: argmax,
  setupFunc: _setup,
);
