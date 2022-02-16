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
// import {RealDiv, RealDivInputs} from '../kernel_names';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {makeTypesMatch} from '../tensor_util';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {floorDiv} from './floorDiv';
// import {op} from './operation';

import '_prelude.dart';
import 'floor_div.dart' show floorDiv;

/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 9, 16]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * ```js
 * // Broadcast div a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(2);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * @param a The first tensor as the numerator.
 * @param b The second tensor as the denominator. Must have the same dtype as
 * `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
T div<T extends Tensor>(Tensor a, Tensor b) {
  return execOp('div', () {
    var $a = convertToTensor(a, 'a', 'div');
    var $b = convertToTensor(b, 'b', 'div');
    final t = makeTypesMatch($a, $b);
    $a = t.first;
    $b = t.second;

    if ($a.dtype == 'int32' && $b.dtype == 'int32') {
      return floorDiv($a, $b);
    }

    final inputs = {'a': $a, 'b': $b};

    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(RealDiv, inputs, {}) as T;
  });
}
