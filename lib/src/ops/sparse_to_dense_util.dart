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
// import {Tensor} from '../tensor';
import '../tensor.dart';

/**
 * Validate sparseToDense inputs.
 *
 * @param sparseIndices A 0-D, 1-D, or 2-D Tensor of type int32.
 * sparseIndices[i] contains the complete index where sparseValues[i] will be
 * placed.
 * @param sparseValues A 0-D or 1-D Tensor. Values
 * corresponding to each row of sparseIndices, or a scalar value to be used for
 * all sparse indices.
 * @param outputShape number[]. Shape of the dense output tensor.
 * @param validateIndices boolean. indice validation is not supported, error
 * will be thrown if it is set.
 */
void validateInput(
  Tensor sparseIndices,
  Tensor sparseValues,
  Shape outputShape,
  Tensor defaultValues,
) {
  if (sparseIndices.dtype != 'int32') {
    throw Exception('tf.sparseToDense() expects the indices to be int32 type,' +
        ' but the dtype was ${sparseIndices.dtype}.');
  }
  if (sparseIndices.rank > 2) {
    throw Exception('sparseIndices should be a scalar, vector, or matrix,' +
        ' but got shape ${sparseIndices.shape}.');
  }

  final numElems = sparseIndices.rank > 0 ? sparseIndices.shape[0] : 1;
  final numDims = sparseIndices.rank > 1 ? sparseIndices.shape[1] : 1;

  if (outputShape.length != numDims) {
    throw Exception('outputShape has incorrect number of elements:,' +
        ' ${outputShape.length}, should be: ${numDims}.');
  }

  final numValues = sparseValues.size;
  if (!(sparseValues.rank == 0 ||
      sparseValues.rank == 1 && numValues == numElems)) {
    throw Exception('sparseValues has incorrect shape ' +
        '${sparseValues.shape}, should be [] or [${numElems}]');
  }

  if (sparseValues.dtype != defaultValues.dtype) {
    throw Exception('sparseValues.dtype must match defaultValues.dtype');
  }
}
