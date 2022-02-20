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
function elu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'elu', 'float32');

  const inputs: EluInputs = {x: $x};

  return ENGINE.runKernel(Elu, inputs as {} as NamedTensorMap);
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
function relu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'relu');

  const inputs: ReluInputs = {x: $x};

  return ENGINE.runKernel(Relu, inputs as {} as NamedTensorMap);
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
function selu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'selu');

  const inputs: SeluInputs = {x: $x};

  return ENGINE.runKernel(Selu, inputs as {} as NamedTensorMap);
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
function sigmoid_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'sigmoid', 'float32');

  const inputs: SigmoidInputs = {x: $x};

  return ENGINE.runKernel(Sigmoid, inputs as {} as NamedTensorMap);
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
function softplus_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'softplus');

  const inputs: SoftplusInputs = {x: $x};
  return ENGINE.runKernel(Softplus, inputs as {} as NamedTensorMap);
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
function relu6_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'relu6');

  const inputs: Relu6Inputs = {x: $x};

  return ENGINE.runKernel(Relu6, inputs as {} as NamedTensorMap);
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
function leakyRelu_<T extends Tensor>(x: T|TensorLike, alpha = 0.2): T {
  const $x = convertToTensor(x, 'x', 'leakyRelu');

  const inputs: LeakyReluInputs = {x: $x};
  const attrs: LeakyReluAttrs = {alpha};

  return ENGINE.runKernel(
      LeakyRelu, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
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
function prelu_<T extends Tensor>(x: T|TensorLike, alpha: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'prelu');
  const $alpha = convertToTensor(alpha, 'alpha', 'prelu');

  const inputs: PreluInputs = {x: $x, alpha: $alpha};
  return ENGINE.runKernel(Prelu, inputs as {} as NamedTensorMap);
}