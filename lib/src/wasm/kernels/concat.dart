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

// import {backend_util, Concat, ConcatAttrs, ConcatInputs, KernelConfig, KernelFunc, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {concatImplCPU} from '../kernel_utils/shared';
// import {identity} from './Identity';
// import {reshape} from './Reshape';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' show SliceList;

import '../kernel_utils/shared.dart' show concatImplCPU, ValueWithShape;
import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'identity.dart';
import 'reshape.dart';

TensorInfo concat({
  required ConcatInputs inputs,
  required BackendWasm backend,
  Map<String, Object?>? attrs,
}) {
  final _axis = attrs!['axis'];
  final axis = util.parseAxisParam(
      _axis is int ? [_axis] : _axis as List<int>, inputs[0].shape)[0];

  var outShape =
      backend_util.computeOutShape(inputs.map((t) => t.shape).toList(), axis);

  // Keep only non-empty tensors (ignore tensors with 0 in their shape).
  final $inputs = inputs.where((t) => util.sizeFromShape(t.shape) > 0).toList();
  if ($inputs.length == 1) {
    return identity(inputs: {'x': $inputs[0]}, backend: backend);
  }

  final out = backend.makeOutput(outShape, inputs[0].dtype);

  if (util.sizeFromShape(outShape) == 0) {
    return out;
  }

  final shapes = $inputs.map((t) => t.shape).toList();
  backend_util.assertParamsConsistent(shapes, axis);

  if ($inputs[0].dtype == 'string') {
    // Any concat of n-dimensional tensors across any axis can be reduced to
    // a concatenation of two-dimensional tensors across the axis 1 by first
    // partitioning the axes of the original tensors into those less than the
    // axis to be concatenated and the rest. Then reshape the tensors
    // into a two-dimensional tensor by collapsing these two sets of axes and
    // concatenate the resulting matrices across the axis 1, finally reshaping
    // the result to have the proper shape.
    final inputs2D = $inputs.map((t) {
      final innerSize = util.sizeFromShape(t.shape.sublistRelaxed(axis));
      final shape = [-1, innerSize];
      return reshape(
          inputs: {'x': t}, backend: backend, attrs: {'shape': shape});
    }).toList();

    final inputsValShapes = inputs2D.map((t) {
      return ValueWithShape(vals: backend.readSync(t.dataId), shape: t.shape);
    }).toList();

    // Concats 2d tensors along axis=1.
    outShape = backend_util.computeOutShape(
        inputs2D.map((t) => t.shape).toList(), 1 /* axis */);
    final simplyConcat = inputs2D[0].shape[0] == 1;
    final outVals = concatImplCPU(
      inputsValShapes,
      outShape,
      inputs[0].dtype,
      simplyConcat,
    ) as List<String>;

    final finalOutShape = backend_util.computeOutShape(
        $inputs.map((t) => t.shape).toList(), axis);

    // out.shape = finalOutShape;
    final outData = backend.dataIdMap.get(out.dataId);
    outData.stringBytes = backend_util.fromStringArrayToUint8(outVals);

    inputs2D.forEach((t) => backend.disposeData(t.dataId));

    return TensorInfo(
      dataId: out.dataId,
      dtype: out.dtype,
      shape: finalOutShape,
    );
  }

  final batchDim = util.sizeFromShape($inputs[0].shape.sublistRelaxed(0, axis));
  int sumInnerDims = 0;
  final innerDims = $inputs.map((input) {
    final innerDim = util.sizeFromShape(input.shape.sublistRelaxed(axis));
    sumInnerDims += innerDim;
    return innerDim;
  }).toList();
  final inVals =
      $inputs.map((input) => backend.typedArrayFromHeap(input)).toList();
  final outVals = backend.typedArrayFromHeap(out);
  for (int b = 0; b < batchDim; b++) {
    int outOffset = b * sumInnerDims;
    for (int i = 0; i < inVals.length; i++) {
      final innerDim = innerDims[i];
      final inOffset = b * innerDim;
      final vals = inVals[i].subarray(inOffset, inOffset + innerDim);
      outVals.set(vals, outOffset);
      outOffset += innerDim;
    }
  }
  return out;
}

final concatConfig = KernelConfigG<BackendWasm, NamedAttrMap>(
  kernelName: Concat,
  backendName: 'wasm',
  kernelFunc: ({required inputs, required backend, attrs}) => concat(
    inputs: List.generate(inputs.length, (index) => inputs[index.toString()]!),
    backend: backend,
    attrs: attrs,
  ),
);
