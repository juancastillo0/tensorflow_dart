/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// import {KernelConfig, KernelFunc, TensorInfo, TopK, TopKAttrs, TopKInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {CppDType} from './types';

import 'dart:typed_data';

import '_prelude.dart';

late final Function(List) _wasmTopK;
// (xId: number, xShapeBytes: Uint8Array, xShapeLength: number,
// xDtype: CppDType, k: number, sorted: boolean, outValuesId: number,
// outIndicesId: number) => void;

void _setup(BackendWasm backend) {
  _wasmTopK = backend.wasm.cwrap(TopK, null /* void */, [
    'number', // xId
    'array', // x.shape
    'number', // x.shape.length
    'number', // x.dtype
    'number', // k
    'bool', // sorted
    'number', // outValuesId
    'number', // outIndicesId
  ]);
}

TensorInfoList topk({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final k = attrs!['k'] as int;
  final sorted = attrs['sorted'] as bool;

  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final xShapeBytes = Uint8List.view(Int32List.fromList(x.shape).buffer);
  final outputShape = [...x.shape];
  outputShape[outputShape.length - 1] = k;
  final outValues = backend.makeOutput(outputShape, x.dtype);
  final outValuesId = backend.dataIdMap.get(outValues.dataId)!.id;
  final outIndices = backend.makeOutput(outputShape, 'int32');
  final outIndicesId = backend.dataIdMap.get(outIndices.dataId)!.id;

  _wasmTopK([
    xId,
    xShapeBytes,
    x.shape.length,
    CppDType.values.byName(x.dtype).index,
    k,
    sorted,
    outValuesId,
    outIndicesId
  ]);

  return TensorInfoList([outValues, outIndices]);
}

final topKConfig = KernelConfigG(
  kernelName: TopK,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: topk,
);
