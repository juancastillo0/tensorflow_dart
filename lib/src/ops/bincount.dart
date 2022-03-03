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
// import {Bincount, BincountAttrs, BincountInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor1D} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';
// import * as util from '../util';

// import {op} from './operation';

import '../util_base.dart' as util;
import '_prelude.dart';

/**
 * Outputs a vector with length `size` and the same dtype as `weights`.
 *
 * If `weights` are empty, then index `i` stores the number of times the value
 * `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the
 * sum of the value in `weights` at each index where the corresponding value in
 * `x` is `i`.
 *
 * Values in `x` outside of the range [0, size) are ignored.
 *
 * @param x The input int tensor, rank 1.
 * @param weights The weights tensor, must have the same shape as x, or a
 *     length-0 Tensor, in which case it acts as all weights equal to 1.
 * @param size Non-negative integer.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
T bincount<T extends Tensor1D>(T x, T weights, int size) {
  return execOp('bincount', () {
    final $x = convertToTensor(x, 'x', 'bincount');
    final $weights = convertToTensor(weights, 'weights', 'bincount');

    util.assert_(
        $x.dtype == 'int32',
        () =>
            'Error in bincount: input ' +
            'dtype must be int32, but got ${$x.dtype}');
    util.assert_(
        size >= 0, () => 'size must be non-negative, but got ${size}.');
    util.assert_(
        $weights.size == $x.size || $weights.size == 0,
        () =>
            'Error in bincount: weights must have the same size as input or' +
            '0-length, but got input shape: ${$x.shape}, weights shape: ' +
            '${$weights.shape}.');

    final inputs = {'x': $x, 'weights': $weights}; // : BincountInputs
    final attrs = {'size': size}; // : BincountAttrs

    return ENGINE.runKernel(Bincount, inputs, attrs) as T;
  });
}
