import 'package:tensorflow_wasm/src/ops/broadcast_util.dart';

import '_prelude.dart';

/**
 * Returns the truth value of `a AND b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalAnd(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
T logicalAnd<T extends Tensor>(Tensor a, Tensor b) {
  return execOpBinary('logicalAnd', LogicalAnd, a, b,
      parseAsDtype: 'string_or_numeric');
}

/**
 * Returns the truth value of `a OR b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalOr(b).print();
 * ```
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
T logicalOr<T extends Tensor>(Tensor a, Tensor b) {
  return execOpBinary('logicalOr', LogicalOr, a, b,
      parseAsDtype: 'string_or_numeric');
}

/**
 * Returns the truth value of `NOT x` element-wise.
 *
 * ```js
 * const a = tf.tensor1d([false, true], 'bool');
 *
 * a.logicalNot().print();
 * ```
 *
 * @param x The input tensor. Must be of dtype 'bool'.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
T logicalNot<T extends Tensor>(T x) {
  return execOp('logicalNot', () {
    final $x = convertToTensor(x, 'x', 'logicalNot', 'bool');
    final inputs = {'x': $x};
    return ENGINE.runKernel(LogicalNot, inputs) as T;
  });
}

/**
 * Returns the truth value of `a XOR b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalXor(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
T logicalXor<T extends Tensor>(Tensor a, Tensor b) {
  const name = 'logicalXor';
  return execOp(name, () {
    final $a = convertToTensor(a, 'a', name, 'bool');
    final $b = convertToTensor(b, 'b', name, 'bool');
    assertAndGetBroadcastShape($a.shape, $b.shape);

    // x ^ y = (x | y) & ~(x & y)
    return logicalAnd(logicalOr($a, $b), logicalNot(logicalAnd($a, $b)));
  });
}
