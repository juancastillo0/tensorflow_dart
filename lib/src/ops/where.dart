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

// import {ENGINE} from '../engine';
// import {Select, SelectInputs} from '../kernel_names';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {broadcastTo} from './broadcast_to';
// import {assertAndGetBroadcastShape} from './broadcast_util';
// import {op} from './operation';

import '_prelude.dart';
import 'broadcast_to.dart';
import 'broadcast_util.dart';

/**
 * Returns the elements, either `a` or `b` depending on the `condition`.
 *
 * If the condition is true, select from `a`, otherwise select from `b`.
 *
 * ```js
 * const cond = tf.tensor1d([false, false, true], 'bool');
 * const a = tf.tensor1d([1 , 2, 3]);
 * const b = tf.tensor1d([-1, -2, -3]);
 *
 * a.where(cond, b).print();
 * ```
 *
 * @param condition The input condition. Must be of dtype bool.
 * @param a If `condition` is rank 1, `a` may have a higher rank but
 *     its first dimension must match the size of `condition`.
 * @param b A tensor with the same dtype as `a` and with shape that is
 *     compatible with `a`.
 * @return A tensor with same dtype as `a` and `b`, and shape that is
 *     broadcastable from `a` and `b`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
T where<T extends Tensor>(Tensor condition, T a, T b) {
  return execOp('where', () {
    final $a = convertToTensor(a, 'a', 'where');
    final $b = convertToTensor(b, 'b', 'where');
    final $condition = convertToTensor(condition, 'condition', 'where', 'bool');
    // TODO: move this logic to forward function when the broadcastTo op is
    // implemented in WASM.
    // Find the broadcastable shape for $condition, $a, and $b.
    final broadcastShape = assertAndGetBroadcastShape(
        assertAndGetBroadcastShape($condition.shape, $a.shape), $b.shape);
    final $broadcastedCondition = broadcastTo($condition, broadcastShape);
    final $broadcastedA = broadcastTo($a, broadcastShape);
    final $broadcastedB = broadcastTo($b, broadcastShape);

    final inputs = {
      'condition': $broadcastedCondition,
      't': $broadcastedA,
      'e': $broadcastedB
    };
    return ENGINE.runKernel(Select, inputs) as T;
  });
}
