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

// import {backend_util, GatherV2, GatherV2Attrs, GatherV2Inputs, KernelConfig, KernelFunc, Tensor, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {reshape} from './Reshape';
// import {CppDType} from './types';

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;
import '_prelude.dart';
import 'reshape.dart';

late final Function(List) _wasmGather;
// (
//     xId: number, dtype: CppDType, xStrides: Uint8Array, stridesSize: number,
//     indicesId: number, batchSize: number, outStrides: Uint8Array,
//     outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmGather = backend.wasm.cwrap('Gather', null /*void*/, [
    'number', // xId
    'number', // dtype
    'array', // xStrides
    'number', // stridesSize
    'number', // indicesId
    'number', // batchSize
    'array', // outStrides
    'number' // outId
  ]);
}

TensorInfo gatherV2({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final indices = inputs['indices']!;
  final axis = [attrs!['axis']! as int];
  final batchDims = attrs['batchDims']! as int;

  // Throw error when any index is out of bound.
  final parsedAxis = util.parseAxisParam(axis, x.shape)[0];
  final indicesVals = backend.readSync(indices.dataId);
  final axisDim = x.shape[parsedAxis];
  for (int i = 0; i < indicesVals.length; ++i) {
    final index = indicesVals[i];
    util.assert_(
        index <= axisDim - 1 && index >= 0,
        () =>
            'GatherV2: the index value ${index} is not in [0, ${axisDim - 1}]');
  }

  final shapeInfo = backend_util.collectGatherOpShapeInfo(
      x as Tensor, indices as Tensor, parsedAxis, batchDims);

  final flattenX = reshape(inputs: {
    'x': x
  }, attrs: {
    'shape': [
      shapeInfo.batchSize,
      shapeInfo.outerSize,
      shapeInfo.dimSize,
      shapeInfo.sliceSize
    ]
  }, backend: backend);
  final indicesSize = util.sizeFromShape(indices.shape);
  final flattenIndex = reshape(inputs: {
    'x': indices
  }, attrs: {
    'shape': [shapeInfo.batchSize, indicesSize / shapeInfo.batchSize]
  }, backend: backend);
  final flattenOutputShape = [
    shapeInfo.batchSize,
    shapeInfo.outerSize,
    indicesSize ~/ shapeInfo.batchSize,
    shapeInfo.sliceSize
  ];

  final out = backend.makeOutput(flattenOutputShape, x.dtype);
  if (util.sizeFromShape(x.shape) == 0) {
    return out;
  }
  final stridesSize = flattenX.shape.length - 1;

  final xData = backend.dataIdMap.get(flattenX.dataId)!;
  final xId = xData.id;

  final indicesData = backend.dataIdMap.get(flattenIndex.dataId)!;
  final indicesId = indicesData.id;

  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final xStridesBytes = Uint8List.view(
      Int32List.fromList(util.computeStrides(flattenX.shape)).buffer);
  final outStridesBytes = Uint8List.view(
      Int32List.fromList(util.computeStrides(flattenOutputShape)).buffer);

  _wasmGather([
    xId,
    CppDType.values.byName(x.dtype).index,
    xStridesBytes,
    stridesSize,
    indicesId,
    shapeInfo.batchSize,
    outStridesBytes,
    outId
  ]);

  backend.disposeData(flattenX.dataId);
  backend.disposeData(flattenIndex.dataId);

  // reshape
  return copyTensorInfo(out, shape: shapeInfo.outputShape);
}

final gatherV2Config = KernelConfigG(
  kernelName: GatherV2,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: gatherV2,
);
