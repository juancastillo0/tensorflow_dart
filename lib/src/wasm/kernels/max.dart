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

// import {backend_util, KernelConfig, KernelFunc, Max, MaxAttrs, MaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {permuteAxesAndTranspose} from './kernel_utils';
// import {CppDType} from './types';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'kernel_utils.dart';

late final Function(List)
    _wasmMax; // : (xId: number, dtype: number, reduceSize: number, outId: number) => void;

void _setupMax(BackendWasm backend) {
  _wasmMax = backend.wasm.cwrap(Max, null /*void*/, [
    'number', // x_id
    'number', // dtype
    'number', // reduce_size
    'number', // out_id
  ]);
}

late final Function(List)
    _wasmMin; // : (xId: number, dtype: number, reduceSize: number, outId: number) => void;

void _setupMin(BackendWasm backend) {
  _wasmMin = backend.wasm.cwrap(Min, null /*void*/, [
    'number', // x_id
    'number', // dtype
    'number', // reduce_size
    'number', // out_id
  ]);
}

ListOrVal<TensorInfo> max({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final axis = (attrs!['reductionIndices'] is int
      ? [attrs['reductionIndices']]
      : attrs['reductionIndices']) as List<int>;
  final keepDims = attrs['keepDims'] as bool;
  return _reduction(
    axis: axis,
    backend: backend,
    keepDims: keepDims,
    opName: 'max',
    wasmFunc: _wasmMax,
    x: inputs['x']!,
  );
}

ListOrVal<TensorInfo> min({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final axis =
      (attrs!['axis'] is int ? [attrs['axis']] : attrs['axis']) as List<int>;
  final keepDims = attrs['keepDims'] as bool;
  return _reduction(
    axis: axis,
    backend: backend,
    keepDims: keepDims,
    opName: 'min',
    wasmFunc: _wasmMin,
    x: inputs['x']!,
  );
}

// function max(args: {backend: BackendWasm, inputs: MaxInputs, attrs: MaxAttrs}): TensorInfo {
ListOrVal<TensorInfo> _reduction({
  required TensorInfo x,
  required String opName,
  required BackendWasm backend,
  required List<int> axis,
  required bool keepDims,
  required dynamic Function(List) wasmFunc,
}) {
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  var inputId = xId;
  var input = x;

  final _p = permuteAxesAndTranspose(x, axis, backend);
  final transposed = _p.transposed;
  final axes = _p.axes;
  final originalAxes = _p.originalAxes;
  final inputWasTransposed = _p.inputWasTransposed;

  if (inputWasTransposed) {
    final transposedId = backend.dataIdMap.get(transposed!.dataId)!.id;
    input = transposed;
    inputId = transposedId;
  }

  final inputRank = input.shape.length;
  backend_util.assertAxesAreInnerMostDims(opName, axes, inputRank);
  final _shapes = backend_util.computeOutAndReduceShapes(input.shape, axes);
  final outShape = _shapes.outShape;
  final reduceShape = _shapes.reduceShape;
  final reduceSize = util.sizeFromShape(reduceShape);

  final out = backend.makeOutput(outShape, x.dtype);
  if (util.sizeFromShape(input.shape) != 0) {
    final outId = backend.dataIdMap.get(out.dataId)!.id;
    wasmFunc([
      inputId,
      CppDType.values.byName(x.dtype).index,
      reduceSize,
      outId,
    ]);
  }

  if (inputWasTransposed) {
    // dispose of the transposed tensor.
    backend.disposeData(transposed!.dataId);
  }

  if (keepDims) {
    // reshape
    final newShape = backend_util.expandShapeToKeepDim(out.shape, originalAxes);
    out.shape = newShape;
  }

  return ListOrVal.val(out);
}

final maxConfig = KernelConfigG(
  kernelName: Max,
  backendName: 'wasm',
  setupFunc: _setupMax,
  kernelFunc: max,
);

final minConfig = KernelConfigG(
  kernelName: Max,
  backendName: 'wasm',
  setupFunc: _setupMin,
  kernelFunc: min,
);
