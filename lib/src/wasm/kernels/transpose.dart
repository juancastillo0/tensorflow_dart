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

// import {KernelConfig, KernelFunc, TensorInfo, Transpose, TransposeAttrs, TransposeInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {identity} from './Identity';
// import {CppDType} from './types';

import 'dart:typed_data';

import '_prelude.dart';
import 'identity.dart';

late final Function(List) _wasmTranspose;
// : ( xId: number, xShape: Uint8Array, xShapeLength: number, dtype: CppDType,
//     outId: number, perm: Uint8Array, permLength: number) => void;

void _setup(BackendWasm backend) {
  _wasmTranspose = backend.wasm.cwrap(Transpose, null /* void */, [
    'number', // xId
    'array', // x.shape
    'number', // x.shape.length
    'number', // dtype
    'number', // outId
    'array', // perm
    'number', // perm.length
  ]);
}

ListOrVal<TensorInfo> transpose({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final xInput = inputs['x']!;
  final permAttr = attrs!['perm'] as Shape;
  // Reduce any dimensions with size one. Lower-rank transpose kernel performs
  // better due to simpler memory access pattern.
  final _reduced = _removeOneSizeDims(xInput.shape, permAttr);
  final perm = _reduced.perm;

  bool permIsNoOp = true;
  for (int i = 0; i < perm.length; i++) {
    if (perm[i] != i) {
      permIsNoOp = false;
    }
  }
  final outShape = _computeOutShape(xInput.shape, permAttr);
  final x = TensorInfo(
      dataId: xInput.dataId, shape: _reduced.shape, dtype: xInput.dtype);

  if (permIsNoOp) {
    final cloned = identity(inputs: inputs, backend: backend);
    cloned.shape = outShape;
    return cloned;
  }

  final out = backend.makeOutput(outShape, x.dtype);
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  final permBytes = Uint8List.view(Int32List.fromList(perm).buffer);
  final xShapeBytes = Uint8List.view(Int32List.fromList(x.shape).buffer);

  _wasmTranspose([
    xId,
    xShapeBytes,
    x.shape.length,
    CppDType.values.byName(x.dtype).index,
    outId,
    permBytes,
    perm.length
  ]);
  return ListOrVal.val(out);
}

Shape _computeOutShape(Shape inShape, Shape perm) {
  final outShape = List.generate(inShape.length, (i) => inShape[perm[i]]);
  return outShape;
}

class _ReducedShape {
  final Shape shape;
  final Shape perm;

  _ReducedShape({
    required this.shape,
    required this.perm,
  });
}

_ReducedShape _removeOneSizeDims(Shape shape, Shape perm) {
  final Shape newShape = [];
  final Shape newPerm = [];
  for (int i = 0; i < shape.length; ++i) {
    if (shape[i] != 1) {
      newShape.add(shape[i]);
    }
    if (shape[perm[i]] != 1) {
      newPerm.add(perm[i]);
    }
  }
  for (int i = 0; i < newPerm.length; ++i) {
    int minValIdx = -1;
    for (int j = 0; j < newPerm.length; ++j) {
      if (newPerm[j] >= i &&
          (minValIdx == -1 || newPerm[minValIdx] > newPerm[j])) {
        minValIdx = j;
      }
    }
    newPerm[minValIdx] = i;
  }
  return _ReducedShape(shape: newShape, perm: newPerm);
}

final transposeConfig = KernelConfigG(
  kernelName: Transpose,
  backendName: 'wasm',
  kernelFunc: transpose,
  setupFunc: _setup,
);
