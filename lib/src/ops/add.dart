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
import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/kernel_names.dart';
import 'package:tensorflow_wasm/src/ops/operation.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/tensor_util.dart';
import 'package:tensorflow_wasm/src/tensor_util_env.dart';

// import {ENGINE} from '../engine';
// import {Add, AddInputs} from '../kernel_names';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {makeTypesMatch} from '../tensor_util';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {op} from './operation';

/**
 * Adds two `tf.Tensor`s element-wise, A + B. Supports broadcasting.
 *
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.tensor1d([10, 20, 30, 40]);
 *
 * a.add(b).print();  // or tf.add(a, b)
 * ```
 *
 * ```js
 * // Broadcast add a with b.
 * const a = tf.scalar(5);
 * const b = tf.tensor1d([10, 20, 30, 40]);
 *
 * a.add(b).print();  // or tf.add(a, b)
 * ```
 * @param a The first `tf.Tensor` to add.
 * @param b The second `tf.Tensor` to add. Must have the same type as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
T add<T extends Tensor>(Tensor a, Tensor b) {
  return execOp('add', () {
    var $a = convertToTensor(a, 'a', 'add');
    var $b = convertToTensor(b, 'b', 'add');
    final p = makeTypesMatch($a, $b);
    $a = p.first;
    $b = p.second;

    final inputs = {'a': $a, 'b': $b}; // BinaryInputs

    return ENGINE.runKernel(Add, inputs) as T;
  });
}

// final add = op({'add_': _add});
