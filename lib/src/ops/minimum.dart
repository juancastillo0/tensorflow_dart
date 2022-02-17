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
// import {Minimum, MinimumInputs} from '../kernel_names';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {makeTypesMatch} from '../tensor_util';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {assertAndGetBroadcastShape} from './broadcast_util';
// import {cast} from './cast';
// import {op} from './operation';

import 'package:tensorflow_wasm/src/ops/broadcast_util.dart';

import '_prelude.dart';
import 'cast.dart' show cast;

/**
 * Returns the min of a and b (`a < b ? a : b`) element-wise.
 * Supports broadcasting.
 *
 * We also expose `minimumStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 3, 16]);
 * const b = tf.tensor1d([1, 2, 9, 4]);
 *
 * a.minimum(b).print();  // or tf.minimum(a, b)
 * ```
 *
 * ```js
 * // Broadcast minimum a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(5);
 *
 * a.minimum(b).print();  // or tf.minimum(a, b)
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same type as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
T minimum<T extends Tensor>(Tensor a, Tensor b) {
  return execOp('minimum', () {
    var $a = convertToTensor(a, 'a', 'minimum');
    var $b = convertToTensor(b, 'b', 'minimum');
    final t = makeTypesMatch($a, $b);
    $a = t.first;
    $b = t.second;
    if ($a.dtype == 'bool') {
      $a = cast($a, 'int32');
      $b = cast($b, 'int32');
    }
    assertAndGetBroadcastShape($a.shape, $b.shape);

    final inputs = {'a': $a, 'b': $b};

    return ENGINE.runKernel(Minimum, inputs) as T;
  });
}
