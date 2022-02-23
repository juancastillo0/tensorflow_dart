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

// import {gather_util, GatherNd, GatherNdInputs, KernelConfig, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/ops/gather_nd_util.dart';
import '_prelude.dart';

late final Function(List) _wasmGatherNd;
// : (
//     xId: number, dtype: CppDType, indicesId: number, numSlices: number,
//     sliceRank: number, sliceSize: number, strides: Uint8Array, outId: number) =>
//     void;

void _setup(BackendWasm backend) {
  _wasmGatherNd = backend.wasm.cwrap(GatherNd, null /*void*/, [
    'number', // xId
    'number', // dtype
    'number', // indicesId
    'number', // numSlices
    'number', // sliceRank
    'number', // sliceSize
    'array', // strides
    'number' // outId
  ]);
}

TensorInfo gatherNd({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final params = inputs['params']!;
  final indices = inputs['indices']!;

  final _gatherInfo = GatherUtil.prepareAndValidate(params, indices);

  final out = backend.makeOutput(_gatherInfo.resultShape, params.dtype);
  if (_gatherInfo.numSlices == 0) {
    return out;
  }

  final indicesShape = indices.shape;
  final sliceRank = indicesShape[indicesShape.length - 1];

  final xData = backend.dataIdMap.get(params.dataId)!;
  final xId = xData.id;
  final indicesData = backend.dataIdMap.get(indices.dataId)!;
  final indicesId = indicesData.id;

  final stridesBytes =
      Uint8List.view(Int32List.fromList(_gatherInfo.strides).buffer);

  final outId = backend.dataIdMap.get(out.dataId)!.id;
  _wasmGatherNd([
    xId,
    CppDType.values.byName(params.dtype).index,
    indicesId,
    _gatherInfo.numSlices,
    sliceRank,
    _gatherInfo.sliceSize,
    stridesBytes,
    outId
  ]);

  return out;
}

final gatherNdConfig = KernelConfigG(
  kernelName: GatherNd,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: gatherNd,
);
