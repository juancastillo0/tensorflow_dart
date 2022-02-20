import '_prelude.dart';

/**
 * Computes exponential linear element-wise: `x > 0 ? x : (e ^ x) - 1`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 1, -3, 2]);
 *
 * x.elu().print();  // or tf.elu(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T elu<T extends Tensor>(T x) {
  return execOpUnary('elu', Elu, x, parseAsDtype: 'float32');
}

/**
 * Computes rectified linear element-wise: `max(x, 0)`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.relu().print();  // or tf.relu(x)
 * ```
 * @param x The input tensor. If the dtype is `bool`, the output dtype will be
 *     `int32'.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T relu<T extends Tensor>(T x) {
  return execOpUnary('relu', Relu, x);
}

/**
 * Computes scaled exponential linear element-wise.
 *
 * `x < 0 ? scale * alpha * (exp(x) - 1) : x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.selu().print();  // or tf.selu(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T selu<T extends Tensor>(T x) {
  return execOpUnary('selu', Selu, x);
}

/**
 * Computes sigmoid element-wise, `1 / (1 + exp(-x))`
 *
 * ```js
 * const x = tf.tensor1d([0, -1, 2, -3]);
 *
 * x.sigmoid().print();  // or tf.sigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T sigmoid<T extends Tensor>(T x) {
  return execOpUnary('sigmoid', Sigmoid, x, parseAsDtype: 'float32');
}

/**
 * Computes softplus of the input `tf.Tensor` element-wise: `log(exp(x) + 1)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.softplus().print();  // or tf.softplus(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T softplus<T extends Tensor>(T x) {
  return execOpUnary('softplus', Softplus, x);
}

/**
 * Computes rectified linear 6 element-wise: `min(max(x, 0), 6)`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 8]);
 *
 * x.relu6().print();  // or tf.relu6(x)
 * ```
 * @param x The input tensor. If the dtype is `bool`, the output dtype will be
 *     `int32'.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T relu6<T extends Tensor>(T x) {
  return execOpUnary('relu6', Relu6, x);
}

/**
 * Computes leaky rectified linear element-wise.
 *
 * See
 * [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf](
 *     http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.leakyRelu(0.1).print();  // or tf.leakyRelu(x, 0.1)
 * ```
 * @param x The input tensor.
 * @param alpha The scaling factor for negative values, defaults to 0.2.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T leakyRelu<T extends Tensor>(T x, [double alpha = 0.2]) {
  return execOp('leakyRelu', () {
    final $x = convertToTensor(x, 'x', 'leakyRelu');

    final inputs = {'x': $x}; //: LeakyReluInputs
    final attrs = {'alpha': alpha}; //: LeakyReluAttrs

    return ENGINE.runKernel(LeakyRelu, inputs, attrs) as T;
  });
}

/**
 * Computes leaky rectified linear element-wise with parametric alphas.
 *
 * `x < 0 ? alpha * x : f(x) = x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 * const alpha = tf.scalar(0.1);
 *
 * x.prelu(alpha).print();  // or tf.prelu(x, alpha)
 * ```
 * @param x The input tensor.
 * @param alpha Scaling factor for negative values.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T prelu<T extends Tensor>(T x, T alpha) {
  return execOp('leakyRelu', () {
    final $x = convertToTensor(x, 'x', 'prelu');
    final $alpha = convertToTensor(alpha, 'alpha', 'prelu');

    final inputs = {'x': $x, 'alpha': $alpha}; // : PreluInputs
    return ENGINE.runKernel(Prelu, inputs) as T;
  });
}
