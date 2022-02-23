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

// import {KernelConfig, KernelFunc, scatter_util, ScatterNd, ScatterNdAttrs, ScatterNdInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import 'dart:typed_data';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/ops/scatter_nd_util.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmScatterNd;
// (    indicesId: number, updatesId: number, dtype: CppDType, sliceRank: number,
//     numUpdates: number, sliceSize: number, strides: Uint8Array,
//     outputSize: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmScatterNd = backend.wasm.cwrap(ScatterNd, null /*void*/, [
    'number', // indicesId
    'number', // updatesId
    'number', // dtype
    'number', // sliceRank
    'number', // numUpdates
    'number', // sliceSize
    'array', // strides
    'number', // outputSize
    'number' // outId
  ]);
}

TensorInfo scatterNd({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final indices = inputs['indices']!;
  final updates = inputs['updates']!;
  final shape = attrs!['shape'] as List<int>;

  final out = backend.makeOutput(shape, updates.dtype);
  if (util.sizeFromShape(shape) == 0) {
    return out;
  }

  final _shapes = ScatterUtil.calculateShapes(updates, indices, shape);
  final sliceRank = _shapes.sliceRank;
  final numUpdates = _shapes.numUpdates;
  final sliceSize = _shapes.sliceSize;
  final strides = _shapes.strides;
  final outputSize = _shapes.outputSize;

  final indicesData = backend.dataIdMap.get(indices.dataId)!;
  final indicesId = indicesData.id;

  final updatesData = backend.dataIdMap.get(updates.dataId)!;
  final updatesId = updatesData.id;

  final stridesBytes = Uint8List.view(Int32List.fromList(strides).buffer);

  final outId = backend.dataIdMap.get(out.dataId)!.id;
  _wasmScatterNd([
    indicesId,
    updatesId,
    CppDType.values.byName(updates.dtype).index,
    sliceRank,
    numUpdates,
    sliceSize,
    stridesBytes,
    outputSize,
    outId
  ]);

  return out;
}

final scatterNdConfig = KernelConfigG(
  kernelName: ScatterNd,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: scatterNd,
);
