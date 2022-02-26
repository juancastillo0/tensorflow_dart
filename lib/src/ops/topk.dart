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

// import {ENGINE} from '../engine';
// import {TopK, TopKAttrs, TopKInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {op} from './operation';

import '_prelude.dart';

/**
 * Finds the values and indices of the `k` largest entries along the last
 * dimension.
 *
 * If the input is a vector (rank=1), finds the k largest entries in the vector
 * and outputs their values and indices as vectors. Thus values[j] is the j-th
 * largest entry in input, and its index is indices[j].
 * For higher rank inputs, computes the top k entries along the last dimension.
 *
 * If two elements are equal, the lower-index element appears first.
 *
 * ```js
 * const a = tf.tensor2d([[1, 5], [4, 3]]);
 * const {values, indices} = tf.topk(a);
 * values.print();
 * indices.print();
 * ```
 * @param x 1-D or higher `tf.Tensor` with last dimension being at least `k`.
 * @param k Number of top elements to look for along the last dimension.
 * @param sorted If true, the resulting `k` elements will be sorted by the
 *     values in descending order.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
TopKValues<T> topk<T extends Tensor>(T x, {int k = 1, bool sorted = true}) {
  return execOp('topk', () {
    final $x = convertToTensor(x, 'x', 'topk');
    if ($x.rank == 0) {
      throw Exception('topk() expects the input to be of rank 1 or higher');
    }
    final lastDim = $x.shape[$x.shape.length - 1];

    if (k < 0) {
      throw Exception("'k' passed to topk() must be >= 0 but got ${k}");
    }

    if (k > lastDim) {
      throw Exception(
          "'k' passed to topk() must be <= the last dimension (${lastDim}) " +
              "but got ${k}");
    }

    final inputs = {'x': $x}; // : TopKInputs
    final attrs = {'k': k, 'sorted': sorted}; // : TopKAttrs

    final result = ENGINE.runKernel(TopK, inputs, attrs) as TensorList;

    return TopKValues(values: result[0] as T, indices: result[1] as T);
  });
}

class TopKValues<T extends Tensor> {
  final T values;
  final T indices;

  TopKValues({
    required this.values,
    required this.indices,
  });
}
