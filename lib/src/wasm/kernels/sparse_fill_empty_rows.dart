/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

// import {backend_util, KernelConfig, KernelFunc, SparseFillEmptyRows, SparseFillEmptyRowsInputs, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {slice} from './Slice';

// import {CppDType} from './types';

import 'dart:typed_data';

import '_prelude.dart';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'slice.dart';

late final Function(List) _wasmSparseFillEmptyRows;
// : (
//     indicesId: number, valuesId: number, valuesDType: number,
//     indicesCount: number, denseRows: number, rank: number,
//     defaultValueId: number, outputIndicesId: number, outputValuesId: number,
//     emptyRowIndicatorId: number, reverseIndexMapId: number,
//     exceptionValuesId: number) => number;

void _setup(BackendWasm backend) {
  _wasmSparseFillEmptyRows =
      backend.wasm.cwrap('SparseFillEmptyRows', 'number', [
    'number', // indicesId
    'number', // valuesId
    'number', // valuesDType
    'number', // indicesCount
    'number', // denseRows
    'number', // rank
    'number', // defaultValueId
    'number', // outputIndicesId
    'number', // outputValuesId
    'number', // emptyRowIndicatorId
    'number', // reverseIndexMapId
    'number', // exceptionValuesId
  ]);
}

TensorInfoList sparseFillEmptyRows({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final indices = inputs['indices']!;
  final values = inputs['values']!;
  final denseShape = inputs['denseShape']!;
  final defaultValue = inputs['defaultValue']!;

  final indicesCount = indices.shape[0];
  final rank = indices.shape[1];
  final denseRows = backend.readSync(denseShape.dataId)[0] as int;

  // Set output size to maximum possible and resize later (actual result
  // might be smaller).
  final maxOutputIndicesShape = [indicesCount + denseRows, rank];

  final indicesId = backend.dataIdMap.get(indices.dataId)!.id;
  final valuesId = backend.dataIdMap.get(values.dataId)!.id;
  final defaultValueId = backend.dataIdMap.get(defaultValue.dataId)!.id;

  final outputIndices =
      backend.makeOutput(maxOutputIndicesShape, indices.dtype);
  final outputIndicesId = backend.dataIdMap.get(outputIndices.dataId)!.id;

  final outputValues =
      backend.makeOutput(maxOutputIndicesShape.sublist(0, 1), values.dtype);
  final outputValuesId = backend.dataIdMap.get(outputValues.dataId)!.id;

  final emptyRowIndicator = backend.makeOutput([denseRows], 'bool');
  final emptyRowIndicatorId =
      backend.dataIdMap.get(emptyRowIndicator.dataId)!.id;

  final reverseIndexMap = backend.makeOutput([indicesCount], indices.dtype);
  final reverseIndexMapId = backend.dataIdMap.get(reverseIndexMap.dataId)!.id;

  final exceptionValues = backend.makeOutput([4], 'int32');
  final exceptionValuesId = backend.dataIdMap.get(exceptionValues.dataId)!.id;

  final outputRows = _wasmSparseFillEmptyRows([
    indicesId,
    valuesId,
    CppDType.values.byName(values.dtype).index,
    indicesCount,
    denseRows,
    rank,
    defaultValueId,
    outputIndicesId,
    outputValuesId,
    emptyRowIndicatorId,
    reverseIndexMapId,
    exceptionValuesId
  ]);

  final exceptionValuesArray =
      backend.readSync(exceptionValues.dataId) as Int32List;

  final String exceptionMessage;
  switch (exceptionValuesArray[0]) {
    case 1:
      {
        exceptionMessage =
            backend_util.getSparseFillEmptyRowsIndicesDenseShapeMismatch(
                exceptionValuesArray[1]);
        break;
      }
    case 2:
      {
        exceptionMessage =
            backend_util.getSparseFillEmptyRowsNegativeIndexErrorMessage(
                exceptionValuesArray[1], exceptionValuesArray[2]);
        break;
      }
    case 3:
      exceptionMessage =
          backend_util.getSparseFillEmptyRowsOutOfRangeIndexErrorMessage(
              exceptionValuesArray[1],
              exceptionValuesArray[2],
              exceptionValuesArray[3]);
      break;
    default:
      exceptionMessage = '';
  }

  backend.disposeData(exceptionValues.dataId);
  if (exceptionMessage.isNotEmpty) {
    backend.disposeData(outputIndices.dataId);
    backend.disposeData(outputValues.dataId);
    backend.disposeData(emptyRowIndicator.dataId);
    backend.disposeData(reverseIndexMap.dataId);
    throw Exception(exceptionMessage);
  }

  TensorInfo resizedIndices = outputIndices;
  TensorInfo resizedValues = outputValues;
  // Overestimated output size.
  if (outputRows != maxOutputIndicesShape[0]) {
    resizedIndices = slice(
      inputs: {'x': outputIndices},
      attrs: {
        'begin': 0,
        'size': [outputRows, rank]
      },
      backend: backend,
    );
    resizedValues = slice(
      inputs: {'x': outputValues},
      attrs: {'begin': 0, 'size': outputRows},
      backend: backend,
    );
    backend.disposeData(outputIndices.dataId);
    backend.disposeData(outputValues.dataId);
  }

  return TensorInfoList([
    resizedIndices,
    resizedValues,
    emptyRowIndicator,
    reverseIndexMap,
  ]);
}

final sparseFillEmptyRowsConfig = KernelConfigG(
  kernelName: SparseFillEmptyRows,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: sparseFillEmptyRows,
);
