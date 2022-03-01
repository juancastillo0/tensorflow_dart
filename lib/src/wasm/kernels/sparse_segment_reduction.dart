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

// import {backend_util, SparseSegmentMeanInputs, SparseSegmentSumInputs, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import 'dart:typed_data';

import '_prelude.dart';

import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

late Function(List) _wasmSparseSegmentReduction;
// : (
//     dataId: number, dtype: number, numRow: number, indicesId: number,
//     segmentIdsId: number, outputId: number, exceptionValuesId: number,
//     isMean: boolean, defaultValue: number) => void;

void setupSparseSegmentReduction(BackendWasm backend) {
  _wasmSparseSegmentReduction =
      backend.wasm.cwrap('SparseSegmentReduction', null /*void*/, [
    'number', // dataId
    'number', // dtype
    'number', // numRow
    'number', // indicesId
    'number', // segmentIdsId
    'number', // outputId
    'number', // exceptionValuesId,
    'number', // isMean
    'number', // defaultValue
  ]);
}

TensorInfo sparseSegmentReduction({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  required bool isMean,
}) {
  final data = inputs['data']!;
  final indices = inputs['indices']!;
  final segmentIds = inputs['segmentIds']!;

  final numIndices = indices.shape[0];
  final segmentIdsBack = (backend.readSync(
      segmentIds.dataId, numIndices - 1, numIndices) as Int32List)[0];
  final lastSegmentIdPlusOne = numIndices > 0 ? segmentIdsBack + 1 : 0;
  final outputRows = lastSegmentIdPlusOne;

  if (outputRows < 0) {
    throw (Exception(backend_util
        .getSparseSegmentReductionNegativeSegmentIdsErrorMessage()));
  }

  final outputShape = [...data.shape];
  outputShape[0] = outputRows;

  final dataId = backend.dataIdMap.get(data.dataId)!.id;
  final indicesId = backend.dataIdMap.get(indices.dataId)!.id;
  final segmentIdsId = backend.dataIdMap.get(segmentIds.dataId)!.id;

  final output = backend.makeOutput(outputShape, data.dtype);
  final outputId = backend.dataIdMap.get(output.dataId)!.id;

  final exceptionValues = backend.makeOutput([4], 'int32');
  final exceptionValuesId = backend.dataIdMap.get(exceptionValues.dataId)!.id;

  _wasmSparseSegmentReduction([
    dataId,
    CppDType.values.byName(data.dtype),
    data.shape[0],
    indicesId,
    segmentIdsId,
    outputId,
    exceptionValuesId,
    isMean,
    0
  ]);

  final exceptionValuesArray =
      backend.readSync(exceptionValues.dataId) as Int32List;

  final String exceptionMessage;
  switch (exceptionValuesArray[0]) {
    case 0:
      {
        exceptionMessage = backend_util
            .getSparseSegmentReductionNegativeSegmentIdsErrorMessage();
        break;
      }
    case 1:
      {
        exceptionMessage = backend_util
            .getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage();
        break;
      }
    case 2:
      exceptionMessage =
          backend_util.getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage(
              exceptionValuesArray[1], exceptionValuesArray[2]);
      break;
    case 3:
      exceptionMessage =
          backend_util.getSparseSegmentReductionIndicesOutOfRangeErrorMessage(
              exceptionValuesArray[1],
              exceptionValuesArray[2],
              exceptionValuesArray[3]);
      break;
    default:
      exceptionMessage = '';
  }

  backend.disposeData(exceptionValues.dataId);
  if (exceptionMessage.isNotEmpty) {
    backend.disposeData(output.dataId);
    throw Exception(exceptionMessage);
  }

  return output;
}
