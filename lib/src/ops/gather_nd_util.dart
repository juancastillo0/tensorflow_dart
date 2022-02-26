import 'package:tensorflow_wasm/src/util_base.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

import '../tensor.dart';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
// import {TensorInfo} from '../kernel_registry';
// import {computeStrides, sizeFromShape} from '../util';

class GatherIndicesInfo {
  final List<int> resultShape;
  final int numSlices;
  final int sliceSize;
  final List<int> strides;

  GatherIndicesInfo({
    required this.resultShape,
    required this.numSlices,
    required this.sliceSize,
    required this.strides,
  });
}

class GatherUtil {
  const GatherUtil._();
/**
 * Validate gather nd inputs.
 *
 * @param tensor The tensor contains the source values.
 * @param indices The tensor contains the indices to slice the source.
 *
 * @returns [resultShape, numUpdates, sliceSize, strides]
 */
  static GatherIndicesInfo prepareAndValidate(
      TensorInfo tensor, TensorInfo indices) {
    final tensorRank = tensor.shape.length;
    final indicesRank = indices.shape.length;
    if (tensorRank < 1) {
      throw Exception(
          'tf.gatherND() expects the input to be rank 1 or higher,' +
              ' but the rank was ${tensorRank}.');
    }
    if (indicesRank < 1) {
      throw Exception(
          'tf.gatherND() expects the indices to be rank 1 or higher,' +
              ' but the rank was ${indicesRank}.');
    }
    if (indices.dtype != 'int32') {
      throw Exception('tf.gatherND() expects the indices to be int32 type,' +
          ' but the dtype was ${indices.dtype}.');
    }
    if (indices.shape[indicesRank - 1] > tensorRank) {
      throw Exception(
          'index innermost dimension length must be <= tensor rank; saw: ' +
              '${indices.shape[indicesRank - 1]} vs. ${tensorRank}');
    }

    if (sizeFromShape(tensor.shape) == 0) {
      throw Exception('Requested more than 0 entries, but input is empty.' +
          ' Input shape: ${tensor.shape}.');
    }

    final indicesShape = indices.shape;
    final sliceRank = indicesShape[indicesShape.length - 1];

    // The result shape is
    //   indices.shape[:-1] + params.shape[indices.shape[-1]:]
    int nResult = 1;
    for (int i = 0; i < indicesShape.length - 1; ++i) {
      nResult *= indicesShape[i];
    }

    final inputShape = tensor.shape;

    final resultShape = [...indicesShape];
    resultShape.removeLast();

    int sliceSize = 1;
    for (int i = sliceRank; i < tensorRank; ++i) {
      sliceSize *= inputShape[i];
      resultShape.add(inputShape[i]);
    }

    final strides = [
      ...computeStrides(tensor.shape).map((stride) => stride ~/ sliceSize),
      1
    ].sublistRelaxed(0, sliceRank);

    return GatherIndicesInfo(
      resultShape: resultShape,
      numSlices: nResult,
      sliceSize: sliceSize,
      strides: strides,
    );
  }
}
