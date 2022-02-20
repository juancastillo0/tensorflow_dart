
/**
 * Computes the softmax normalized vector given the logits.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 *
 * a.softmax().print();  // or tf.softmax(a)
 * ```
 *
 * ```js
 * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
 *
 * a.softmax().print();  // or tf.softmax(a)
 * ```
 *
 * @param logits The logits array.
 * @param dim The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function softmax_<T extends Tensor>(logits: T|TensorLike, dim = -1): T {
  const $logits = convertToTensor(logits, 'logits', 'softmax', 'float32');

  if (dim === -1) {
    dim = $logits.rank - 1;
  }
  if (dim !== $logits.rank - 1) {
    throw Error(
        'Softmax along a non-last dimension is not yet supported. ' +
        `Logits was rank ${$logits.rank} and dim was ${dim}`);
  }

  const inputs: SoftmaxInputs = {logits: $logits};
  const attrs: SoftmaxAttrs = {dim};

  return ENGINE.runKernel(
      Softmax, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

/**
 * Computes the log softmax.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * ```js
 * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * @param logits The logits array.
 * @param axis The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function logSoftmax_<T extends Tensor>(logits: T|TensorLike, axis = -1): T {
  const $logits = convertToTensor(logits, 'logits', 'logSoftmax');

  if (axis === -1) {
    axis = $logits.rank - 1;
  }
  if (axis !== $logits.rank - 1) {
    throw Error(
        'Log Softmax along a non-last dimension is not yet supported. ' +
        `Logits was rank ${$logits.rank} and axis was ${axis}`);
  }

  // const forward: ForwardFunc<Tensor> = (backend, save) => {
  //   const keepDims = true;
  //   const xMax = max(logits, axis, true);
  //   const shifted = sub(logits, xMax);
  //   const value =
  //       sub(cast(shifted, 'float32'), log(sum(exp(shifted), axis,
  //       keepDims)));
  //   save([value]);
  //   return value;
  // };

  // Use a custom gradient for numerical stability.
  const customOp = customGrad((logits: Tensor, save: GradSaveFunc) => {
    const keepDims = true;
    const xMax = max(logits, axis, true);
    const shifted = sub(logits, xMax);
    const value =
        sub(cast(shifted, 'float32'), log(sum(exp(shifted), axis, keepDims)));
    save([value]);

    const gradFunc = (dy: Tensor, saved: Tensor[]) => {
      const [value] = saved;
      const keepDims = true;
      const softmax = exp(value);
      return sub(dy, mul(sum(dy, axis, keepDims), softmax));
    };
    return {value, gradFunc};
  });

  return customOp($logits) as T;

  // TODO Use Engine.runKernel when CPU/WebGL/WASM backends implement this.
  // const inputs: LogSoftmaxInputs = {logits: $logits};
  // const attrs: LogSoftmaxAttrs = {axis};
  // return ENGINE.runKernel(
  //            LogSoftmax, inputs as {} as NamedTensorMap,
  //            attrs as {} as NamedAttrMap);
}


/**
 * Normalizes the activation of a local neighborhood across or within
 * channels.
 *
 * @param x The input tensor. The 4-D input tensor is treated as a 3-D array
 *     of 1D vectors (along the last dimension), and each vector is
 *     normalized independently.
 * @param depthRadius The number of adjacent channels in the 1D normalization
 *     window.
 * @param bias A constant bias term for the basis.
 * @param alpha A scale factor, usually positive.
 * @param beta An exponent.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function localResponseNormalization_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, depthRadius = 5, bias = 1, alpha = 1, beta = 0.5): T {
  const $x = convertToTensor(x, 'x', 'localResponseNormalization');
  util.assert(
      $x.rank === 4 || $x.rank === 3,
      () => `Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${$x.rank}.`);
  util.assert(
      util.isInt(depthRadius),
      () => `Error in localResponseNormalization: depthRadius must be an ` +
          `integer but got depthRadius ${depthRadius}.`);

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }

  const inputs: LRNInputs = {x: x4D};

  const attrs: LRNAttrs = {depthRadius, bias, alpha, beta};

  // tslint:disable-next-line: no-unnecessary-type-assertion
  const res = ENGINE.runKernel(
                  LRN, inputs as {} as NamedTensorMap,
                  attrs as {} as NamedAttrMap) as T;

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  } else {
    return res;
  }
}


/**
 * Batch normalization.
 *
 * As described in
 * [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167).
 *
 * Mean, variance, scale, and offset can be of two shapes:
 *   - The same shape as the input.
 *   - In the common case, the depth dimension is the last dimension of x, so
 *     the values would be an `tf.Tensor1D` of shape [depth].
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that parameters passed are of given rank
 *   - `tf.batchNorm2d`
 *   - `tf.batchNorm3d`
 *   - `tf.batchNorm4d`
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function batchNorm_<R extends Rank>(
    x: Tensor<R>|TensorLike, mean: Tensor<R>|Tensor1D|TensorLike,
    variance: Tensor<R>|Tensor1D|TensorLike,
    offset?: Tensor<R>|Tensor1D|TensorLike,
    scale?: Tensor<R>|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor<R> {
  if (varianceEpsilon == null) {
    varianceEpsilon = 0.001;
  }
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor<R>|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor<R>|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }

  util.assert(
      $mean.rank === $variance.rank,
      () => 'Batch normalization gradient requires mean and variance to have ' +
          'equal ranks.');
  util.assert(
      $offset == null || $mean.rank === $offset.rank,
      () => 'Batch normalization gradient requires mean and offset to have ' +
          'equal ranks.');
  util.assert(
      $scale == null || $mean.rank === $scale.rank,
      () => 'Batch normalization gradient requires mean and scale to have ' +
          'equal ranks.');

  const x4D: Tensor4D = xAs4D($x);

  const inputs: FusedBatchNormInputs = {
    x: x4D,
    scale: $scale,
    offset: $offset,
    mean: $mean,
    variance: $variance
  };

  const attrs: FusedBatchNormAttrs = {varianceEpsilon};

  // tslint:disable-next-line: no-unnecessary-type-assertion
  const res = ENGINE.runKernel(
                  FusedBatchNorm, inputs as {} as NamedTensorMap,
                  attrs as {} as NamedAttrMap) as Tensor<R>;

  return reshape(res, $x.shape);
}