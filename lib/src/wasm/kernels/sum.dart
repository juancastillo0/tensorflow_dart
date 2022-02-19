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

// import {backend_util, KernelConfig, KernelFunc, Sum, SumAttrs, SumInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {permuteAxesAndTranspose} from './kernel_utils';
// import {CppDType} from './types';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'kernel_utils.dart';

late final Function(List)
    _wasmSum; //: (xId: number, reduceSize: number, dtype: number, outId: number) => void;

void _setupSum(BackendWasm backend) {
  _wasmSum = backend.wasm.cwrap(Sum, null /*void*/, [
    'number', // input_id
    'number', // reduce_size
    'number', // dtype
    'number', // out_id
  ]);
}

late final Function(List)
    _wasmProd; //: (xId: number, reduceSize: number, dtype: number, outId: number) => void;

void _setupProd(BackendWasm backend) {
  _wasmProd = backend.wasm.cwrap(Prod, null /*void*/, [
    'number', // input_id
    'number', // reduce_size
    'number', // dtype
    'number', // out_id
  ]);
}

ListOrVal<TensorInfo> sum({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  return _reduction(
    backend: backend,
    inputs: inputs,
    opName: 'sum',
    wasmFunc: _wasmSum,
    attrs: attrs!,
  );
}

ListOrVal<TensorInfo> prod({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  return _reduction(
    backend: backend,
    inputs: inputs,
    opName: 'prod',
    wasmFunc: _wasmProd,
    attrs: attrs!,
  );
}

ListOrVal<TensorInfo> _reduction({
  required String opName,
  required dynamic Function(List) wasmFunc,
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  required NamedAttrMap attrs,
}) {
  final axis =
      (attrs['axis'] is int ? [attrs['axis']] : attrs['axis']) as List<int>;
  final keepDims = attrs['keepDims'] as bool;
  final x = inputs['x']!;
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  var inputId = xId;
  var input = x;

  final _p = permuteAxesAndTranspose(x, axis, backend);
  final transposed = _p.transposed;
  final axes = _p.axes;
  final originalAxes = _p.originalAxes;
  final inputWasTransposed = _p.inputWasTransposed;

  var reductionAxes = axes;
  if (inputWasTransposed) {
    final transposedId = backend.dataIdMap.get(transposed!.dataId)!.id;
    if (transposedId != xId) {
      // transpose was not a no-op. We will need to dispose of this
      // once we are done.
      input = transposed;
      inputId = transposedId;
      reductionAxes = backend_util.getInnerMostAxes(
          reductionAxes.length, input.shape.length);
    }
  }

  backend_util.assertAxesAreInnerMostDims(
      opName, reductionAxes, input.shape.length);
  final _shapes =
      backend_util.computeOutAndReduceShapes(input.shape, reductionAxes);
  final outShape = _shapes.outShape;
  final reduceShape = _shapes.reduceShape;
  final reduceSize = util.sizeFromShape(reduceShape);

  final out = backend.makeOutput(outShape, input.dtype);
  if (util.sizeFromShape(input.shape) != 0) {
    final outId = backend.dataIdMap.get(out.dataId)!.id;
    wasmFunc([
      inputId,
      reduceSize,
      CppDType.values.byName(out.dtype).index,
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

  return out;
}

final sumConfig = KernelConfigG(
  kernelName: Sum,
  backendName: 'wasm',
  setupFunc: _setupSum,
  kernelFunc: sum,
);

final prodConfig = KernelConfigG(
  kernelName: Prod,
  backendName: 'wasm',
  setupFunc: _setupProd,
  kernelFunc: prod,
);
