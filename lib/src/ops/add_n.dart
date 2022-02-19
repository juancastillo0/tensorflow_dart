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
// import {AddN, AddNInputs} from '../kernel_names';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';
// import * as util from '../util';

// import {op} from './operation';
import 'package:collection/collection.dart';

import '_prelude.dart';
import '../util_base.dart' as util;

/**
 * Adds a list of `tf.Tensor`s element-wise, each with the same shape and dtype.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 *
 * tf.addN([a, b, c]).print();
 * ```
 * @param tensors A list of tensors with the same shape and dtype.
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
T addN<T extends Tensor>(List<T> tensors) {
  return execOp('addN', () {
    util.assert_(tensors is List,
        () => 'The argument passed to tf.addN() must be a list of tensors');
    util.assert_(
        tensors.length >= 1,
        () =>
            'Must pass at least one tensor to tf.addN(), but got ' +
            '${tensors.length}');

    final $tensors = tensors
        .mapIndexed((i, t) => convertToTensor(t, 'tensors${i}', 'addN'))
        .toList();

    final firstTensor = $tensors[0];
    $tensors.forEach((t) {
      if (t.dtype != firstTensor.dtype) {
        throw Exception(
            'All tensors passed to tf.addN() must have the same dtype');
      }
    });

    $tensors.forEach((t) {
      if (!util.arraysEqual(t.shape, firstTensor.shape)) {
        throw Exception(
            'All tensors passed to tf.addN() must have the same shape');
      }
    });

    final inputs = Map.fromIterables(
        Iterable.generate($tensors.length, (i) => i.toString()),
        $tensors); // AddNInputs

    return ENGINE.runKernel(AddN, inputs) as T;
  });
}
