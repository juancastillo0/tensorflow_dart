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

/**
 * Subtracts two `tf.Tensor`s element-wise, A - B. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([10, 20, 30, 40]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 *
 * a.sub(b).print();  // or tf.sub(a, b)
 * ```
 *
 * ```js
 * // Broadcast subtract a with b.
 * const a = tf.tensor1d([10, 20, 30, 40]);
 * const b = tf.scalar(5);
 *
 * a.sub(b).print();  // or tf.sub(a, b)
 * ```
 * @param a The first `tf.Tensor` to subtract from.
 * @param b The second `tf.Tensor` to be subtracted. Must have the same dtype as
 * `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
T sub<T extends Tensor>(
  Tensor a,
  Tensor b,
) {
  return execOpBinary('sub', Sub, a, b);
}

/**
 * Computes `-1 * x` element-wise.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, -2, 0], [2, 2]);
 *
 * x.neg().print();  // or tf.neg(x)
 * ```
 *
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T neg<T extends Tensor>(T x) {
  return execOp('neg', () {
    final $x = convertToTensor(x, 'x', 'neg');

    final inputs = {'x': $x}; // NegInputs
    return ENGINE.runKernel(Neg, inputs) as T;
  });
}

/**
 * Computes absolute value element-wise: `abs(x)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.abs().print();  // or tf.abs(x)
 * ```
 * @param x The input `tf.Tensor`.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T abs<T extends Tensor>(Tensor x) {
  return execOp('abs', () {
    final $x = convertToTensor(x, 'x', 'abs');

    if ($x.dtype == 'complex64') {
      final inputs = {'x': $x};
      return ENGINE.runKernel(ComplexAbs, inputs) as T;
    } else {
      final inputs = {'x': $x};
      return ENGINE.runKernel(Abs, inputs) as T;
    }
  });
}

/**
 * Returns an element-wise indication of the sign of a number.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3, NaN, 0]);
 *
 * x.sign().print();  // or tf.sign(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T sign<T extends Tensor>(Tensor x) {
  return execOpUnary('sign', Sign, x);
}
