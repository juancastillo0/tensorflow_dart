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

import '_prelude.dart';
import 'package:tensorflow_wasm/src/ops/clone.dart';

// import {ENGINE} from '../engine';
// import {Concat, ConcatAttrs, ConcatInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensorArray} from '../tensor_util_env';
// import {TensorLike} from '../types';
// import {assert} from '../util';

// import {clone} from './clone';
// import {op} from './operation';

/**
 * Concatenates a list of `tf.Tensor`s along a given axis.
 *
 * The tensors ranks and types must match, and their sizes must match in all
 * dimensions except `axis`.
 *
 * Also available are stricter rank-specific methods that assert that
 * `tensors` are of the given rank:
 *   - `tf.concat1d`
 *   - `tf.concat2d`
 *   - `tf.concat3d`
 *   - `tf.concat4d`
 *
 * Except `tf.concat1d` (which does not have axis param), all methods have
 * same signature as this method.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * a.concat(b).print();  // or a.concat(b)
 * ```
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.concat([a, b, c]).print();
 * ```
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [10, 20]]);
 * const b = tf.tensor2d([[3, 4], [30, 40]]);
 * const axis = 1;
 * tf.concat([a, b], axis).print();
 * ```
 * @param tensors A list of tensors to concatenate.
 * @param axis The axis to concate along. Defaults to 0 (the first dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
T concat<T extends Tensor>(List<T> tensors, [int axis = 0]) {
  return execOp('concat', () {
    assert(tensors.length >= 1, () => 'Pass at least one tensor to concat');

    final $tensors =
        convertToTensorArray(tensors, 'tensors', 'concat', 'string_or_numeric');

    if ($tensors[0].dtype == 'complex64') {
      $tensors.forEach((tensor) {
        if (tensor.dtype != 'complex64') {
          throw Exception(
              'Cannot concatenate complex64 tensors with a tensor with dtype ${tensor.dtype}. ');
        }
      });
    }

    if ($tensors.length == 1) {
      return clone($tensors[0]);
    }

    final inputs = Map.fromIterables(
      Iterable.generate($tensors.length, (i) => i.toString()),
      $tensors,
    ); // : ConcatInputs
    final attr = {'axis': axis}; // : ConcatAttrs

    return ENGINE.runKernel(
      Concat,
      inputs,
      attr,
    ) as T;
  });
}
