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

// import {backend_util, KernelConfig, KernelFunc, SparseReshape, SparseReshapeInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import 'dart:typed_data';

import '_prelude.dart';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

late final Function(List) _wasmSparseReshape;
// : (
//     inputIndicesId: number, inputShapeId: number, newShapeId: number,
//     nnz: number, newIndicesId: number, outputShapeId: number,
//     exceptionValuesId: number) => void;

void _setup(BackendWasm backend) {
  _wasmSparseReshape = backend.wasm.cwrap(SparseReshape, null /*void*/, [
    'number', // inputIndicesId
    'number', // inputShapeId
    'number', // newShapeId
    'number', // nnz
    'number', // newIndicesId
    'number', // outputShapeId
    'number', // exceptionValuesId
  ]);
}

TensorInfoList sparseReshape({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final inputIndices = inputs['inputIndices']!;
  final inputShape = inputs['inputShape']!;
  final newShape = inputs['newShape']!;

  if (inputIndices.shape.length != 2) {
    throw Exception(
        'Input indices should be a matrix but received shape ${inputIndices.shape}');
  }
  if (inputShape.shape.length != 1) {
    throw Exception(
        'Input shape should be a vector but received shape ${inputShape.shape}');
  }
  if (newShape.shape.length != 1) {
    throw Exception(
        'Target shape should be a vector but received shape ${newShape.shape}');
  }

  final inputIndicesId = backend.dataIdMap.get(inputIndices.dataId)!.id;
  final inputShapeId = backend.dataIdMap.get(inputShape.dataId)!.id;
  final newShapeId = backend.dataIdMap.get(newShape.dataId)!.id;

  final nnz = inputIndices.shape[0];
  final outputRank = util.sizeFromShape(newShape.shape);

  final newIndices = backend.makeOutput([nnz, outputRank], inputIndices.dtype);
  final newIndicesId = backend.dataIdMap.get(newIndices.dataId)!.id;

  final outputShape = backend.makeOutput([outputRank], newShape.dtype);
  final outputShapeId = backend.dataIdMap.get(outputShape.dataId)!.id;

  final exceptionValues = backend.makeOutput([3], 'int32');
  final exceptionValuesId = backend.dataIdMap.get(exceptionValues.dataId)!.id;

  _wasmSparseReshape([
    inputIndicesId,
    inputShapeId,
    newShapeId,
    nnz,
    newIndicesId,
    outputShapeId,
    exceptionValuesId
  ]);

  final exceptionValuesArray =
      backend.readSync(exceptionValues.dataId) as Int32List;

  final String exceptionMessage;
  switch (exceptionValuesArray[0]) {
    case 0:
      {
        exceptionMessage = backend_util
            .getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(
                exceptionValuesArray[1], exceptionValuesArray[2]);
        break;
      }
    case 1:
      {
        exceptionMessage =
            backend_util.getSparseReshapeNegativeOutputDimErrorMessage(
                exceptionValuesArray[1], exceptionValuesArray[2]);
        break;
      }
    case 2:
      exceptionMessage =
          backend_util.getSparseReshapeEmptyTensorZeroOutputDimErrorMessage();
      break;
    case 3:
      {
        final inputShapeValues =
                (backend.readSync(inputShape.dataId) as Int32List),
            outputShapeValues =
                (backend.readSync(outputShape.dataId) as Int32List);
        exceptionMessage =
            backend_util.getSparseReshapeInputOutputMultipleErrorMessage(
                inputShapeValues, outputShapeValues);
        break;
      }
    case 4:
      {
        final inputShapeValues =
                (backend.readSync(inputShape.dataId) as Int32List),
            outputShapeValues =
                (backend.readSync(outputShape.dataId) as Int32List);
        exceptionMessage =
            backend_util.getSparseReshapeInputOutputMismatchErrorMessage(
                inputShapeValues, outputShapeValues);
        break;
      }
    default:
      exceptionMessage = '';
  }

  backend.disposeData(exceptionValues.dataId);
  if (exceptionMessage.isNotEmpty) {
    backend.disposeData(newIndices.dataId);
    backend.disposeData(outputShape.dataId);
    throw Exception(exceptionMessage);
  }

  return TensorInfoList([newIndices, outputShape]);
}

final sparseReshapeConfig = KernelConfigG(
  kernelName: SparseReshape,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: sparseReshape,
);
