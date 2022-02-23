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
// import {Tensor} from '../tensor';
// import {computeStrides, sizeFromShape} from '../util';

import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

import '../util_base.dart';

class ScatterShapeInfo {
  final int sliceRank;
  final int numUpdates;
  final int sliceSize;
  final List<int> strides;
  final int outputSize;

  ScatterShapeInfo({
    required this.sliceRank,
    required this.numUpdates,
    required this.sliceSize,
    required this.strides,
    required this.outputSize,
  });
}

class ScatterUtil {
  const ScatterUtil._();

/**
 * Check whether updates.shape = indices.shape[:batchDim] +
 * shape[sliceDim:]
 *
 * @param x The input tensor.
 */
  static void validateUpdateShape(
      List<int> shape, Tensor indices, Tensor updates) {
    final sliceDim = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;
    final batchDim = (indices.rank > 1) ? indices.rank - 1 : 1;

    final shapeError = 'Must have updates.shape = indices.shape[:batchDim] + ' +
        'shape[sliceDim:], got updates.shape: ${updates.shape}' +
        ', indices.shape: ${indices.shape}, shape: ${shape}' +
        ', sliceDim: ${sliceDim}, and batchDim: ${batchDim}.';

    if (updates.rank < batchDim) {
      throw Exception(shapeError + ' update.rank < ${batchDim}. ');
    }
    if (shape.length < sliceDim + (updates.rank - batchDim)) {
      throw Exception(shapeError +
          ' Output shape length < ${sliceDim + (updates.rank - batchDim)}');
    }
    if (updates.rank != batchDim + shape.length - sliceDim) {
      throw Exception(
          shapeError + ' update.rank != ${batchDim + shape.length - sliceDim}');
    }
    for (int d = 0; d < batchDim; ++d) {
      if (updates.shape[d] != indices.shape[d]) {
        throw Exception(shapeError +
            ' updates.shape[${d}] (${updates.shape[d]}) != indices.shape[${d}] (${indices.shape[d]}).');
      }
    }
    for (int d = 0; d < updates.rank - batchDim; ++d) {
      if (updates.shape[d + batchDim] != shape[d + sliceDim]) {
        throw Exception(shapeError +
            ' updates.shape[${d + batchDim}] (${updates.shape[d + batchDim]}) != shape[${d + batchDim}] (${shape[d + batchDim]})');
      }
    }
  }

/**
 * Validate scatter nd inputs.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 */
  static void validateInput(Tensor updates, Tensor indices, Shape shape) {
    if (indices.rank < 1) {
      throw Exception(
          'tf.scatterND() expects the indices to be rank 1 or higher,' +
              " but the rank was ${indices.rank}.");
    }
    if (updates.rank < 1) {
      throw Exception(
          'tf.scatterND() expects the updates to be rank 1 or higher,' +
              " but the rank was ${updates.rank}.");
    }
    if (indices.dtype != 'int32') {
      throw Exception(
          "The dtype of 'indices' should be int32, but got dtype: ${indices.dtype}");
    }
    if (shape.length < 1) {
      throw Exception(
          "Output rank must be greater or equal to 1, but got shape: ${shape}");
    }

    if (shape.length == 0) {
      if (indices.size == 0) {
        throw Exception(
            "Indices specified for empty output. indices shape: ${indices.shape}");
      }
      if (updates.size == 0) {
        throw Exception(
            "Updates specified for empty output. updates shape: ${updates.shape}");
      }
    }

    validateUpdateShape(shape, indices, updates);
  }

/**
 * Calculate the shape information for the output.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 *
 * @returns ScatterShapeInfo
 */
  static ScatterShapeInfo calculateShapes(
    TensorInfo updates,
    TensorInfo indices,
    Shape shape,
  ) {
    // Calculate the number of dimensions in indices
    final indicesRank = indices.shape.length;
    final sliceRank = (indicesRank > 1) ? indices.shape[indicesRank - 1] : 1;

    // Calculate the number of elements that make up each slice of our updated
    // tensor. This allows us to work with flattened tensors and copy over whole
    // slices at a time.
    final totalNd = shape.length;

    int sliceSize = 1;
    for (int i = sliceRank; i < totalNd; ++i) {
      sliceSize *= shape[i];
    }

    final safeSliceDim = (sliceRank < 1) ? 1 : sliceRank;
    final numUpdates = sizeFromShape(indices.shape) ~/ safeSliceDim;

    final strides = [...computeStrides(shape.slice(0, sliceRank)), 1];
    final outputSize = sizeFromShape(shape);
    return ScatterShapeInfo(
      sliceRank: sliceRank,
      numUpdates: numUpdates,
      sliceSize: sliceSize,
      strides: strides,
      outputSize: outputSize,
    );
  }
}
