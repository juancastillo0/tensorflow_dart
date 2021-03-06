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
// import {Transpose, TransposeAttrs, TransposeInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';
// import * as util from '../util';

// import {op} from './operation';

import '_prelude.dart';
import '../util_base.dart' as util;
import 'package:collection/collection.dart';

/**
 * Transposes the `tf.Tensor`. Permutes the dimensions according to `perm`.
 *
 * The returned `tf.Tensor`'s dimension `i` will correspond to the input
 * dimension `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`,
 * where `n` is the rank of the input `tf.Tensor`. Hence by default, this
 * operation performs a regular matrix transpose on 2-D input `tf.Tensor`s.
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
 *
 * a.transpose().print();  // or tf.transpose(a)
 * ```
 *
 * @param x The tensor to transpose.
 * @param perm The permutation of the dimensions of a.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
T transpose<T extends Tensor>(T x, {List<int>? perm}) {
  return execOp('transpose', () {
    final $x = convertToTensor(x, 'x', 'transpose');

    perm ??= $x.shape.mapIndexed((i, s) => i).toList().reversed.toList();

    util.assert_(
        $x.rank == perm!.length,
        () =>
            "Error in transpose: rank of input ${$x.rank} " +
            "must match length of perm ${perm}.");
    perm!.forEach((axis) {
      util.assert_(
          axis >= 0 && axis < $x.rank,
          () =>
              "All entries in 'perm' must be between 0 and ${$x.rank - 1}" +
              " but got ${perm}");
    });

    if ($x.rank <= 1) {
      return $x.clone();
    }

    final inputs = {'x': $x}; // : TransposeInputs
    final attrs = {'perm': perm}; // : TransposeAttrs

    return ENGINE.runKernel(Transpose, inputs, attrs) as T;
  });
}
