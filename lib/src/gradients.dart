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

// import {CustomGradientFunc, ENGINE} from './engine';
// import {Scalar, Tensor, Variable} from './tensor';
// import {NamedTensorMap} from './tensor_types';
// import {convertToTensor, convertToTensorArray} from './tensor_util_env';
// import {TensorLike} from './types';
// import * as util from './util';

import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/tensor_util_env.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';
import 'util_base.dart' as util;

/**
 * Provided `f(x)`, returns another function `g(x, dy?)`, which gives the
 * gradient of `f(x)` with respect to `x`.
 *
 * If `dy` is provided, the gradient of `f(x).mul(dy).sum()` with respect to
 * `x` is computed instead. `f(x)` must take a single tensor `x` and return a
 * single tensor `y`. If `f()` takes multiple inputs, use `tf.grads` instead.
 *
 * ```js
 * // f(x) = x ^ 2
 * const f = x => x.square();
 * // f'(x) = 2x
 * const g = tf.grad(f);
 *
 * const x = tf.tensor1d([2, 3]);
 * g(x).print();
 * ```
 *
 * ```js
 * // f(x) = x ^ 3
 * const f = x => x.pow(tf.scalar(3, 'int32'));
 * // f'(x) = 3x ^ 2
 * const g = tf.grad(f);
 * // f''(x) = 6x
 * const gg = tf.grad(g);
 *
 * const x = tf.tensor1d([2, 3]);
 * gg(x).print();
 * ```
 *
 * @param f The function f(x), to compute gradient for.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
Tensor Function(Tensor x, Tensor? dy) grad(Tensor Function(Tensor x) f) {
  util.assert_(
      f is Function, () => 'The f passed in grad(f) must be a function');
  return (x, dy) {
    // x can be of any dtype, thus null as the last argument.
    final $x = convertToTensor(x, 'x', 'tf.grad', 'string_or_numeric');
    final Tensor? $dy =
        (dy != null) ? convertToTensor(dy, 'dy', 'tf.grad') : null;
    return ENGINE.tidy(() {
      final _gradients = ENGINE.gradients(() => f($x), [$x], dy: $dy);
      final grads = _gradients.grads;
      final value = _gradients.value;
      if ($dy != null) {
        util.assertShapesMatch(
            value.shape,
            $dy.shape,
            'The shape of dy passed in grad(f)(x, dy) must match the shape ' +
                'returned by f(x)');
      }
      _checkGrads(grads);
      return grads[0];
    });
  };
}

/**
 * Provided `f(x1, x2,...)`, returns another function `g([x1, x2,...], dy?)`,
 * which gives an array of gradients of `f()` with respect to each input
 * [`x1`,`x2`,...].
 *
 * If `dy` is passed when calling `g()`, the gradient of
 * `f(x1,...).mul(dy).sum()` with respect to each input is computed instead.
 * The provided `f` must take one or more tensors and return a single tensor
 * `y`. If `f()` takes a single input, we recommend using `tf.grad` instead.
 *
 * ```js
 * // f(a, b) = a * b
 * const f = (a, b) => a.mul(b);
 * // df / da = b, df / db = a
 * const g = tf.grads(f);
 *
 * const a = tf.tensor1d([2, 3]);
 * const b = tf.tensor1d([-2, -3]);
 * const [da, db] = g([a, b]);
 * console.log('da');
 * da.print();
 * console.log('db');
 * db.print();
 * ```
 *
 * @param f The function `f(x1, x2,...)` to compute gradients for.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
List<Tensor> Function(List<Tensor> args, Tensor? dy) grads(
  Tensor Function(List<Tensor>) f,
) {
  util.assert_(
      f is Function, () => 'The f passed in grads(f) must be a function');
  return (args, dy) {
    util.assert_(
        args is List,
        () =>
            'The args passed in grads(f)(args) must be an array ' +
            'of `Tensor`s or `TensorLike`s');
    // args can be of any dtype, thus null as the last argument.
    final $args =
        convertToTensorArray(args, 'args', 'tf.grads', 'string_or_numeric');
    final Tensor? $dy =
        (dy != null) ? convertToTensor(dy, 'dy', 'tf.grads') : null;
    return ENGINE.tidy(() {
      final _g = ENGINE.gradients(() => f($args), $args, dy: $dy);
      if ($dy != null) {
        util.assertShapesMatch(
            _g.value.shape,
            $dy.shape,
            'The shape of dy passed in grads(f)([x1,...], dy) must ' +
                'match the shape returned by f([x1,...])');
      }
      _checkGrads(_g.grads);
      return _g.grads;
    });
  };
}

/**
 * Like `tf.grad`, but also returns the value of `f()`. Useful when `f()`
 * returns a metric you want to show.
 *
 * The result is a rich object with the following properties:
 * - grad: The gradient of `f(x)` w.r.t `x` (result of `tf.grad`).
 * - value: The value returned by `f(x)`.
 *
 * ```js
 * // f(x) = x ^ 2
 * const f = x => x.square();
 * // f'(x) = 2x
 * const g = tf.valueAndGrad(f);
 *
 * const x = tf.tensor1d([2, 3]);
 * const {value, grad} = g(x);
 *
 * console.log('value');
 * value.print();
 * console.log('grad');
 * grad.print();
 * ```
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
// function valueAndGrad<I extends Tensor, O extends Tensor>(f: (x: I) => O): (
//     x: I, dy?: O) => {
//   value: O;
//   grad: I;
// } {
//   util.assert_(
//       util.isFunction(f),
//       () => 'The f passed in valueAndGrad(f) must be a function');
//   return (x: I, dy?: O) => {
//     util.assert_(
//         x instanceof Tensor,
//         () => 'The x passed in valueAndGrad(f)(x) must be a tensor');
//     util.assert_(
//         dy == null || dy instanceof Tensor,
//         () => 'The dy passed in valueAndGrad(f)(x, dy) must be a tensor');
//     const {grads, value} = ENGINE.gradients(() => f(x), [x], dy);
//     checkGrads(grads);
//     return {grad: grads[0] as I, value};
//   };
// }

/**
 * Like `tf.grads`, but returns also the value of `f()`. Useful when `f()`
 * returns a metric you want to show.
 *
 * The result is a rich object with the following properties:
 * - grads: The gradients of `f()` w.r.t each input (result of `tf.grads`).
 * - value: The value returned by `f(x)`.
 *
 * ```js
 * // f(a, b) = a * b
 * const f = (a, b) => a.mul(b);
 * // df/da = b, df/db = a
 * const g = tf.valueAndGrads(f);
 *
 * const a = tf.tensor1d([2, 3]);
 * const b = tf.tensor1d([-2, -3]);
 * const {value, grads} = g([a, b]);
 *
 * const [da, db] = grads;
 *
 * console.log('value');
 * value.print();
 *
 * console.log('da');
 * da.print();
 * console.log('db');
 * db.print();
 * ```
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
WithGradients<O> Function(List<Tensor> args, O? dy)
    valueAndGrads<O extends Tensor>(O Function(List<Tensor>) f) {
  util.assert_(f is Function,
      () => 'The f passed in valueAndGrads(f) must be a function');
  return (args, dy) {
    util.assert_(
        args is List && args.every((arg) => arg is Tensor),
        () =>
            'The args passed in valueAndGrads(f)(args) must be array of ' +
            'tensors');
    util.assert_(dy == null || dy is Tensor,
        () => 'The dy passed in valueAndGrads(f)(args, dy) must be a tensor');
    final res = ENGINE.gradients(() => f(args), args, dy: dy);
    if (dy != null) {
      util.assertShapesMatch(
          res.value.shape,
          dy.shape,
          'The shape of dy passed in valueAndGrads(f)([x1,...], dy) must ' +
              'match the shape returned by f([x1,...])');
    }
    _checkGrads(res.grads);
    return res;
  };
}

class ValueGrad<V, G> {
  final V value;
  final G grads;

  ValueGrad({
    required this.value,
    required this.grads,
  });
}

/**
 * Computes and returns the gradient of f(x) with respect to the list of
 * trainable variables provided by `varList`. If no list is provided, it
 * defaults to all trainable variables.
 *
 * ```js
 * const a = tf.variable(tf.tensor1d([3, 4]));
 * const b = tf.variable(tf.tensor1d([5, 6]));
 * const x = tf.tensor1d([1, 2]);
 *
 * // f(a, b) = a * x ^ 2 + b * x
 * const f = () => a.mul(x.square()).add(b.mul(x)).sum();
 * // df/da = x ^ 2, df/db = x
 * const {value, grads} = tf.variableGrads(f);
 *
 * Object.keys(grads).forEach(varName => grads[varName].print());
 * ```
 *
 * @param f The function to execute. f() should return a scalar.
 * @param varList The list of variables to compute the gradients with respect
 *     to. Defaults to all trainable variables.
 * @returns An object with the following keys and values:
 *   - `value`: The value of the function `f`.
 *   - `grads`: A map from the names of the variables to the gradients.
 *     If the `varList` argument is provided explicitly and contains a subset of
 *     non-trainable variables, this map in the return value will contain keys
 *     that map the names of the non-trainable variables to `null`.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
ValueGrad<Tensor, Map<String, Tensor?>> variableGrads(
  // TODO: Scalar
  Tensor Function() f,
  List<Variable>? varList,
) {
  util.assert_(f is Function,
      () => 'The f passed in variableGrads(f) must be a function');
  util.assert_(
      varList == null || varList is List && varList.every((v) => v is Variable),
      () =>
          'The varList passed in variableGrads(f, varList) must be an array ' +
          'of variables');

  final specifiedVarList = varList != null;
  if (!specifiedVarList) {
    // Get all of the trainable variables.
    varList = [...ENGINE.registeredVariables.values];
  }

  final specifiedNonTrainable = specifiedVarList
      ? varList.where((variable) => !variable.trainable)
      : null;

  // Prune non-trainable variables.
  final originalVarCount = varList.length;
  varList = varList.where((variable) => variable.trainable).toList();
  util.assert_(
      varList.length > 0,
      () =>
          'variableGrads() expects at least one of the input variables to ' +
          'be trainable, but none of the ${originalVarCount} variables is ' +
          'trainable.');

  final allowNoGradients = true;
  final _g = ENGINE.gradients<Tensor>(f, varList,
      dy: null, allowNoGradients: allowNoGradients);
  final grads = _g.grads;

  util.assert_(
      grads.any((g) => g != null),
      () =>
          'Cannot find a connection between any variable and the result of ' +
          'the loss function y=f(x). Please make sure the operations that ' +
          'use variables are inside the function f passed to minimize().');
  util.assert_(
      _g.value.rank == 0,
      () =>
          'The f passed in variableGrads(f) must return a scalar, but it ' +
          'returned a rank-${_g.value.rank} tensor');

  final Map<String, Tensor?> namedGrads = {};
  int __i = 0;
  varList.forEach((v) {
    if (grads[__i++] != null) {
      namedGrads[v.name] = grads[__i];
    }
  });
  if (specifiedNonTrainable != null) {
    // If varList is explicitly provided and contains non-trainable values,
    // add them to the returned gradients with `null` values.
    specifiedNonTrainable.forEach((v) => namedGrads[v.name] = null);
  }
  return ValueGrad(value: _g.value, grads: namedGrads);
}

/**
 * Overrides the gradient computation of a function `f`.
 *
 * Takes a function
 * `f(...inputs, save) => {value: Tensor, gradFunc: (dy, saved) => Tensor[]}`
 * and returns another function `g(...inputs)` which takes the same inputs as
 * `f`. When called, `g` returns `f().value`. In backward mode, custom gradients
 * with respect to each input of `f` are computed using `f().gradFunc`.
 *
 * The `save` function passsed to `f` should be used for saving tensors needed
 * in the gradient. And the `saved` passed to the `gradFunc` is a
 * `NamedTensorMap`, which contains those saved tensor.
 *
 * ```js
 * const customOp = tf.customGrad((x, save) => {
 *   // Save x to make sure it's available later for the gradient.
 *   save([x]);
 *   // Override gradient of our custom x ^ 2 op to be dy * abs(x);
 *   return {
 *     value: x.square(),
 *     // Note `saved.x` which points to the `x` we saved earlier.
 *     gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
 *   };
 * });
 *
 * const x = tf.tensor1d([-1, -2, 3]);
 * const dx = tf.grad(x => customOp(x));
 *
 * console.log(`f(x):`);
 * customOp(x).print();
 * console.log(`f'(x):`);
 * dx(x).print();
 * ```
 *
 * @param f The function to evaluate in forward mode, which should return
 *     `{value: Tensor, gradFunc: (dy, saved) => Tensor[]}`, where `gradFunc`
 *     returns the custom gradients of `f` with respect to its inputs.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
T Function(List<Tensor>) customGrad<T extends Tensor>(CustomGradientFunc<T> f) {
  return ENGINE.customGrad(f);
}

void _checkGrads(List<Tensor> grads) {
  final numNullGradients = grads.where((g) => g == null).length;
  if (numNullGradients > 0) {
    throw Exception(
        'Cannot compute gradient of y=f(x) with respect to x. Make sure that '
        'the f you passed encloses all operations that lead from x to y.');
  }
}
