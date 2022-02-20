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

// import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, broadcast_util, KernelConfig, KernelFunc, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {reshape} from './Reshape';

import 'dart:typed_data';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import '../../ops/broadcast_util.dart' as broadcast_util;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' show SliceList;

import 'reshape.dart';
import 'dart:math' as math;

late final Function(List) _wasmBatchMatMul;
// : (    aId: number, aShape: Uint8Array, aShapeSize: number, bId: number,
//     bShape: Uint8Array, bShapeSize: number, transposeA: boolean,
//     transposeB: boolean, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmBatchMatMul = backend.wasm.cwrap(BatchMatMul, null /* void */, [
    'number', // a_id
    'array', // a_shape
    'number', // a_shape.length
    'number', // b_id
    'array', // b_shape
    'number', // b_shape.length
    'number', // transpose_a
    'number', // transpose_b
    'number' // out_id
  ]);
}

ListOrVal<TensorInfo> batchMatMul({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final a = inputs['a']!;
  final b = inputs['b']!;
  final transposeA = attrs!['transposeA'] as bool;
  final transposeB = attrs['transposeB'] as bool;

  if (a.dtype != 'float32' || b.dtype != 'float32') {
    throw Exception(
        'BatchMatMul for non non-float32 tensors not yet supported.');
  }

  final aRank = a.shape.length;
  final bRank = b.shape.length;

  final innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
  final innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];

  final outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
  final outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];

  final outerDimsA = a.shape.slice(0, -2);
  final outerDimsB = b.shape.slice(0, -2);

  final batchDimA = util.sizeFromShape(outerDimsA);
  final batchDimB = util.sizeFromShape(outerDimsB);

  final outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(
      a.shape.slice(0, -2), b.shape.slice(0, -2));
  final outShape = [...outShapeOuterDims, outerShapeA, outerShapeB];

  util.assert_(
      innerShapeA == innerShapeB,
      () =>
          'Error in matMul: inner shapes (${innerShapeA}) and (' +
          '${innerShapeB}) of Tensors with shapes ${a.shape} and ' +
          '${b.shape} and transposeA=${transposeA}' +
          ' and transposeB=${transposeB} must match.');

  final a3dShape = transposeA
      ? [batchDimA, innerShapeA, outerShapeA]
      : [batchDimA, outerShapeA, innerShapeA];
  final b3dShape = transposeB
      ? [batchDimB, outerShapeB, innerShapeB]
      : [batchDimB, innerShapeB, outerShapeB];

  // The rest of the implementation is designed to operate on rank-3 tensors
  final a3d =
      reshape(inputs: {'x': a}, backend: backend, attrs: {'shape': a3dShape})
          .asVal!;
  final b3d =
      reshape(inputs: {'x': b}, backend: backend, attrs: {'shape': b3dShape})
          .asVal!;

  final a3dId = backend.dataIdMap.get(a3d.dataId)!.id;
  final b3dId = backend.dataIdMap.get(b3d.dataId)!.id;

  final leftDim = transposeA ? a3d.shape[2] : a3d.shape[1];
  final rightDim = transposeB ? b3d.shape[1] : b3d.shape[2];
  final batchDim = math.max(batchDimA, batchDimB);

  final out = backend.makeOutput([batchDim, leftDim, rightDim], a3d.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final aShapeBytes = Uint8List.view(Int32List.fromList(a3d.shape).buffer);
  final bShapeBytes = Uint8List.view(Int32List.fromList(b3d.shape).buffer);

  _wasmBatchMatMul([
    a3dId,
    aShapeBytes,
    a3d.shape.length,
    b3dId,
    bShapeBytes,
    b3d.shape.length,
    transposeA,
    transposeB,
    outId
  ]);

  backend.disposeData(a3d.dataId);
  backend.disposeData(b3d.dataId);

  out.shape = outShape;
  return ListOrVal.val(out);
}

final batchMatMulConfig = KernelConfigG(
  kernelName: BatchMatMul,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: batchMatMul,
);
