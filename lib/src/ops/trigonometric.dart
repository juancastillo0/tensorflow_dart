import '_prelude.dart';

/**
 * Computes sin of the input Tensor element-wise: `sin(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.sin().print();  // or tf.sin(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function sin_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'sin', 'float32');

  const inputs: SinInputs = {x: $x};

  return ENGINE.runKernel(Sin, inputs as {} as NamedTensorMap);
}
export const sin = op({sin_});


/**
 * Computes hyperbolic sin of the input `tf.Tensor` element-wise: `sinh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.sinh().print();  // or tf.sinh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function sinh_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'sinh');
  const inputs: SinhInputs = {x: $x};

  return ENGINE.runKernel(Sinh, inputs as {} as NamedTensorMap);
}
export const sinh = op({sinh_});



/**
 * Computes asin of the input `tf.Tensor` element-wise: `asin(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.asin().print();  // or tf.asin(x)
 * ```
 * @param x The input tensor.
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function asin_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'asin');
  const inputs: AsinInputs = {x: $x};

  return ENGINE.runKernel(Asin, inputs as {} as NamedTensorMap);
}
export const asin = op({asin_});



/**
 * Computes inverse hyperbolic sin of the input `tf.Tensor` element-wise:
 * `asinh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.asinh().print();  // or tf.asinh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function asinh_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'asinh');

  const inputs: AsinhInputs = {x: $x};

  return ENGINE.runKernel(Asinh, inputs as {} as NamedTensorMap);
}
export const asinh = op({asinh_});


/**
 * Computes cos of the input `tf.Tensor` element-wise: `cos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.cos().print();  // or tf.cos(x)
 * ```
 * @param x The input tensor. Must be float32 type.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function cos_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'cos', 'float32');

  const inputs: CosInputs = {x: $x};

  return ENGINE.runKernel(Cos, inputs as {} as NamedTensorMap);
}
export const cos = op({cos_});


/**
 * Computes hyperbolic cos of the input `tf.Tensor` element-wise: `cosh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.cosh().print();  // or tf.cosh(x)
 * ```
 * @param x The input tensor. Must be float32 type.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function cosh_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'cosh', 'float32');
  const inputs: CoshInputs = {x: $x};

  return ENGINE.runKernel(Cosh, inputs as {} as NamedTensorMap);
}
export const cosh = op({cosh_});


/**
 * Computes acos of the input `tf.Tensor` element-wise: `acos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.acos().print();  // or tf.acos(x)
 * ```
 * @param x The input tensor.
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function acos_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'acos');
  const inputs: AcosInputs = {x: $x};

  return ENGINE.runKernel(Acos, inputs as {} as NamedTensorMap);
}
export const acos = op({acos_});


/**
 * Computes the inverse hyperbolic cos of the input `tf.Tensor` element-wise:
 * `acosh(x)`
 *
 * ```js
 * const x = tf.tensor1d([10, 1, 3, 5.7]);
 *
 * x.acosh().print();  // or tf.acosh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function acosh_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'acosh');
  const inputs: AcoshInputs = {x: $x};

  return ENGINE.runKernel(Acosh, inputs as {} as NamedTensorMap);
}
export const acosh = op({acosh_});


/**
 * Computes tan of the input `tf.Tensor` element-wise, `tan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.tan().print();  // or tf.tan(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function tan_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'tan', 'float32');

  const inputs: TanInputs = {x: $x};

  return ENGINE.runKernel(Tan, inputs as {} as NamedTensorMap);
}
export const tan = op({tan_});


/**
 * Computes hyperbolic tangent of the input `tf.Tensor` element-wise: `tanh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, 70]);
 *
 * x.tanh().print();  // or tf.tanh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function tanh_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'tanh', 'float32');

  const inputs: TanhInputs = {x: $x};

  return ENGINE.runKernel(Tanh, inputs as {} as NamedTensorMap);
}
export const tanh = op({tanh_});


/**
 * Computes atan of the input `tf.Tensor` element-wise: `atan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.atan().print();  // or tf.atan(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function atan_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'atan');

  const inputs: AtanInputs = {x: $x};

  return ENGINE.runKernel(Atan, inputs as {} as NamedTensorMap);
}
export const atan = op({atan_});


/**
 * Computes arctangent of `tf.Tensor`s a / b element-wise: `atan2(a, b)`.
 * Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1.0, 1.0, -1.0, .7]);
 * const b = tf.tensor1d([2.0, 13.0, 3.5, .21]);
 *
 * tf.atan2(a, b).print()
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function atan2_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'atan2');
  let $b = convertToTensor(b, 'b', 'atan2');
  [$a, $b] = makeTypesMatch($a, $b);

  const inputs: Atan2Inputs = {a: $a, b: $b};

  return ENGINE.runKernel(Atan2, inputs as {} as NamedTensorMap);
}

export const atan2 = op({atan2_});


/**
 * Computes inverse hyperbolic tan of the input `tf.Tensor` element-wise:
 * `atanh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, .1, -.1, .7]);
 *
 * x.atanh().print();  // or tf.atanh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function atanh_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'atanh');

  const inputs: AtanhInputs = {x: $x};

  return ENGINE.runKernel(Atanh, inputs as {} as NamedTensorMap);
}
export const atanh = op({atanh_});