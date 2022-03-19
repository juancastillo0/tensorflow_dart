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

// import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs, broadcast_util, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {FusableActivation} from './types';

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/ops/fused_types.dart';
import 'package:collection/collection.dart' hide ListExtensions;
import '_prelude.dart';
import '../../ops/broadcast_util.dart' as broadcast_util;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' show SliceList;

late final Function(List) _wasmFusedMatMul;
// (aId: number, aShape: Uint8Array, aShapeSize: number, bId: number,
//  bShape: Uint8Array, bShapeSize: number, transposeA: boolean,
//  transposeB: boolean, activation: number, biasId: number,
//  preluActivationWeightsId: number, leakyreluAlpha: number, outId: number) =>
//     void;

void _setup(BackendWasm backend) {
  _wasmFusedMatMul = backend.wasm.cwrap(FusedMatMul_, null /* void */, [
    'number', // a_id
    'array', // a_shape
    'number', // a_shape.length
    'number', // b_id
    'array', // b_shape
    'number', // b_shape.length
    'number', // transpose_a
    'number', // transpose_b
    'number', // activation
    'number', // biasId
    'number', // preluActivationWeightsId
    'number', // leakyreluAlpha
    'number' // out_id
  ]);
}

TensorInfo fusedBatchMatMul(
    {required NamedTensorInfoMap inputs,
    required BackendWasm backend,
    NamedAttrMap? attrs}) {
  final a = inputs['a']!;
  final b = inputs['b']!;
  final bias = inputs['bias'];
  final preluActivationWeights = inputs['preluActivationWeights'];

  if (a.dtype != 'float32' || b.dtype != 'float32') {
    throw Exception(
        "_FusedMatMul for non non-float32 tensors not yet supported.");
  }

  final transposeA = attrs!['transposeA'] as bool;
  final transposeB = attrs['transposeB'] as bool;
  final activation = attrs['activation'] as Activation;
  final leakyreluAlpha = attrs['leakyreluAlpha'] as num?;
  final aId = backend.dataIdMap.get(a.dataId)!.id;
  final bId = backend.dataIdMap.get(b.dataId)!.id;

  int biasId = 0;
  if (bias != null) {
    final biasData = backend.dataIdMap.get(bias.dataId)!;
    if (biasData.shape.length != 1) {
      throw Exception("_FusedMatMul only supports rank-1 bias but got " +
          "rank ${biasData.shape.length}.");
    }
    biasId = biasData.id;
  }
  final preluActivationWeightsId = preluActivationWeights == null
      ? 0
      : backend.dataIdMap.get(preluActivationWeights.dataId)!.id;
  final fusedActivation = FusableActivation.values
      .firstWhereOrNull((a) => a.name == activation.name);
  if (fusedActivation == null) {
    throw Exception(
        "${activation} activation not yet supported for FusedConv2D " +
            "in the wasm backend.");
  }

  final leftDim = transposeA ? a.shape[2] : a.shape[1];
  final rightDim = transposeB ? b.shape[1] : b.shape[2];
  final batchDims = broadcast_util.assertAndGetBroadcastShape(
      a.shape.sublistRelaxed(0, -2), b.shape.sublistRelaxed(0, -2));

  final out = backend.makeOutput([...batchDims, leftDim, rightDim], a.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final aShapeBytes = Uint8List.view(Int32List.fromList(a.shape).buffer);
  final bShapeBytes = Uint8List.view(Int32List.fromList(b.shape).buffer);

  _wasmFusedMatMul([
    aId,
    aShapeBytes,
    a.shape.length,
    bId,
    bShapeBytes,
    b.shape.length,
    transposeA,
    transposeB,
    fusedActivation.index,
    biasId,
    preluActivationWeightsId,
    leakyreluAlpha ?? 0,
    outId
  ]);

  return out;
}

final fusedMatMulConfig_ = KernelConfigG(
  kernelName: FusedMatMul_,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: fusedBatchMatMul,
);
