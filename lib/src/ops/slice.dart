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

import '_prelude.dart';

// import {ENGINE} from '../engine';
// import {Slice, SliceAttrs, SliceInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {Rank, TensorLike} from '../types';

// import {op} from './operation';

/**
 * Extracts a slice from a `tf.Tensor` starting at coordinates `begin`
 * and is of size `size`.
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that `x` is of the given rank:
 *   - `tf.slice1d`
 *   - `tf.slice2d`
 *   - `tf.slice3d`
 *   - `tf.slice4d`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.slice([1], [2]).print();
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * x.slice([1, 0], [1, 2]).print();
 * ```
 * @param x The input `tf.Tensor` to slice from.
 * @param begin The coordinates to start the slice from. The length can be
 *     less than the rank of x - the rest of the axes will have implicit 0 as
 *     start. Can also be a single number, in which case it specifies the
 *     first axis.
 * @param size The size of the slice. The length can be less than the rank of
 *     x - the rest of the axes will have implicit -1. A value of -1 requests
 *     the rest of the dimensions in the axis. Can also be a single number,
 *     in which case it specifies the size of the first axis.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
T slice<R extends Rank, T extends Tensor<R>>(
  T x,
  // : number|number[]
  List<int> begin,
  // : number|number[]
  List<int>? size,
) {
  return execOp('slice', () {
    final $x = convertToTensor(x, 'x', 'slice', 'string_or_numeric');

    if ($x.rank == 0) {
      throw Exception('Slicing scalar is not possible');
    }

    final inputs = {'x': $x}; // : SliceInputs
    final attrs = {'begin': begin, 'size': size}; // : SliceAttrs

    return ENGINE.runKernel(
      Slice,
      inputs,
      attrs,
    ) as T;
  });
}

/**
 * Extracts a strided slice of a tensor.
 *
 * Roughly speaking, this op extracts a slice of size (end-begin)/stride from
 * the given input tensor (x). Starting at the location specified by begin the
 * slice continues by adding stride to the index until all dimensions are not
 * less than end. Note that a stride can be negative, which causes a reverse
 * slice.
 *
 * ```js
 * const t = tf.tensor3d([1, 1, 1 ,2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
 *    [3, 2, 3]);
 * t.stridedSlice([1, 0, 0], [2, 1, 3], [1, 1, 1]).print()  // [[[3, 3, 3]]]
 * t.stridedSlice([1, 0, 0], [2, 2, 3], [1, 1, 1]).print()  // [[[3, 3, 3],
 *                                                     // [4, 4, 4]]]
 * t.stridedSlice([1, -1, 0], [2, -3, 3], [1, -1, 1]).print() // [[[4, 4, 4],
 *                                                     // [3, 3, 3]]]
 * ```
 *
 * @param x The tensor to stride slice.
 * @param begin The coordinates to start the slice from.
 * @param end: The coordinates to end the slice at.
 * @param strides: The size of the slice.
 * @param beginMask: If the ith bit of beginMask is set, begin[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param endMask: If the ith bit of endMask is set, end[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param shrinkAxisMask: a bitmask where bit i implies that
 * the ith specification should shrink the dimensionality. begin and end must
 * imply a slice of size 1 in the dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
Tensor stridedSlice(
  Tensor x,
  List<int> begin,
  List<int> end, {
  List<int>? strides,
  int beginMask = 0,
  int endMask = 0,
  int ellipsisMask = 0,
  int newAxisMask = 0,
  int shrinkAxisMask = 0,
}) {
  return execOp('stridedSlice', () {
    final $x = convertToTensor(x, 'x', 'stridedSlice', 'string_or_numeric');

    final inputs = {'x': $x}; // StridedSliceInputs
    final attrs = StridedSliceAttrs(
      begin: begin,
      end: end,
      strides: strides,
      beginMask: beginMask,
      endMask: endMask,
      ellipsisMask: ellipsisMask,
      newAxisMask: newAxisMask,
      shrinkAxisMask: shrinkAxisMask,
    );

    return ENGINE.runKernel(StridedSlice, inputs, attrs) as Tensor;
  });
}

/**
 * Splits a `tf.Tensor` into sub tensors.
 *
 * If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
 * into `numOrSizeSplits` smaller tensors.
 * Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
 *
 * If `numOrSizeSplits` is a number array, splits `x` into
 * `numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
 * same size as `x` except along dimension `axis` where the size is
 * `numOrSizeSplits[i]`.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
 * const [a, b] = tf.split(x, 2, 1);
 * a.print();
 * b.print();
 *
 * const [c, d, e] = tf.split(x, [1, 2, 1], 1);
 * c.print();
 * d.print();
 * e.print();
 * ```
 *
 * @param x The input tensor to split.
 * @param numOrSizeSplits Either an integer indicating the number of
 * splits along the axis or an array of integers containing the sizes of
 * each output tensor along the axis. If a number then it must evenly divide
 * `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
 * Can contain one -1 indicating that dimension is to be inferred.
 * @param axis The dimension along which to split. Defaults to 0 (the first
 * dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
TensorList split<T extends Tensor>(
  Tensor x,
  // : number[]|number
  List<int> numOrSizeSplits, {
  int axis = 0,
}) {
  return execOp('split', () {
    final $x = convertToTensor(x, 'x', 'split');

    final inputs = {'x': $x}; // : SplitVInputs
    final attr = {
      'numOrSizeSplits': numOrSizeSplits,
      'axis': axis
    }; // : SplitVAttrs

    return ENGINE.runKernel(SplitV, inputs, attr) as TensorList;
  });
}
