

/**
 * Creates a `tf.Tensor` filled with a scalar value.
 *
 * ```js
 * tf.fill([2, 2], 4).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param value The scalar value to fill the tensor with.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 * 'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function fill<R extends Rank>(
    shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
  const attrs: FillAttrs = {shape, value, dtype};

  return ENGINE.runKernel(Fill, {}, attrs as {} as NamedAttrMap);
}

export {fill};


/**
 * Return an evenly spaced sequence of numbers over the given interval.
 *
 * ```js
 * tf.linspace(0, 9, 10).print();
 * ```
 * @param start The start value of the sequence.
 * @param stop The end value of the sequence.
 * @param num The number of values to generate.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function linspace(start: number, stop: number, num: number): Tensor1D {
  if (num <= 0) {
    throw new Error('The number of values should be positive.');
  }

  const attrs: LinSpaceAttrs = {start, stop, num};
  return ENGINE.runKernel(LinSpace, {}, attrs as {} as NamedAttrMap);
}


/**
 * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
 *
 * The tensor is a is half-open interval meaning it includes start, but
 * excludes stop. Decrementing ranges and negative step values are also
 * supported.sv
 *
 *
 * ```js
 * tf.range(0, 9, 2).print();
 * ```
 *
 * @param start An integer start value
 * @param stop An integer stop value
 * @param step An integer increment (will default to 1 or -1)
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function range(
    start: number, stop: number, step = 1,
    dtype: 'float32'|'int32' = 'float32'): Tensor1D {
  if (step === 0) {
    throw new Error('Cannot have a step of zero');
  }

  const attrs: RangeAttrs = {start, stop, step, dtype};

  return ENGINE.runKernel(Range, {} /* inputs */, attrs as {} as NamedAttrMap);
}



/**
 * Creates a one-hot `tf.Tensor`. The locations represented by `indices` take
 * value `onValue` (defaults to 1), while all other locations take value
 * `offValue` (defaults to 0). If `indices` is rank `R`, the output has rank
 * `R+1` with the last axis of size `depth`. 
 * `indices` used to encode prediction class must start from 0. For example,
 *  if you have 3 classes of data, class 1 should be encoded as 0, class 2
 *  should be 1, and class 3 should be 2. 
 *
 * ```js
 * tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
 * ```
 *
 * @param indices `tf.Tensor` of indices with dtype `int32`. Indices must 
 * start from 0.
 * @param depth The depth of the one hot dimension.
 * @param onValue A number used to fill in the output when the index matches
 * the location.
 * @param offValue A number used to fill in the output when the index does
 *     not match the location.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function oneHot_(
    indices: Tensor|TensorLike, depth: number, onValue = 1,
    offValue = 0): Tensor {
  if (depth < 2) {
    throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
  }
  const $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');

  const inputs: OneHotInputs = {indices: $indices};
  const attrs: OneHotAttrs = {depth, onValue, offValue};

  return ENGINE.runKernel(
      OneHot, inputs as unknown as NamedTensorMap,
      attrs as unknown as NamedAttrMap);
}

export const oneHot = op({oneHot_});


/**
 * Creates a `tf.Tensor` with all elements set to 1.
 *
 * ```js
 * tf.ones([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 *     'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function ones<R extends Rank>(
    shape: ShapeMap[R], dtype: DataType = 'float32'): Tensor<R> {
  if (dtype === 'complex64') {
    const real = ones(shape, 'float32');
    const imag = zeros(shape, 'float32');
    return complex(real, imag);
  }
  const values = makeOnesTypedArray(sizeFromShape(shape), dtype);
  return ENGINE.makeTensor(values, shape, dtype) as Tensor<R>;
}


/**
 * Creates a `tf.Tensor` with all elements set to 1 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.onesLike(x).print();
 * ```
 * @param x A tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function onesLike_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'onesLike');

  const inputs: OnesLikeInputs = {x: $x};
  return ENGINE.runKernel(OnesLike, inputs as {} as NamedTensorMap);
}

export const onesLike = op({onesLike_});


/**
 * Creates a `tf.Tensor` with all elements set to 0.
 *
 * ```js
 * tf.zeros([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Can
 *     be 'float32', 'int32' or 'bool'. Defaults to 'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function zeros<R extends Rank>(
    shape: ShapeMap[R], dtype: DataType = 'float32'): Tensor<R> {
  if (dtype === 'complex64') {
    const real = zeros(shape, 'float32');
    const imag = zeros(shape, 'float32');
    return complex(real, imag);
  }
  const values = makeZerosTypedArray(sizeFromShape(shape), dtype);
  return ENGINE.makeTensor(values, shape, dtype) as Tensor<R>;
}


/**
 * Creates a `tf.Tensor` with all elements set to 0 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.zerosLike(x).print();
 * ```
 *
 * @param x The tensor of required shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function zerosLike_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'zerosLike');
  const inputs: ZerosLikeInputs = {x: $x};
  return ENGINE.runKernel(ZerosLike, inputs as {} as NamedTensorMap);
}
export const zerosLike = op({zerosLike_});
