
/**
 * Creates a `tf.Tensor` with values drawn from a multinomial distribution.
 *
 * ```js
 * const probs = tf.tensor([.75, .25]);
 * tf.multinomial(probs, 3).print();
 * ```
 *
 * @param logits 1D array with unnormalized log-probabilities, or
 *     2D array of shape `[batchSize, numOutcomes]`. See the `normalized`
 *     parameter.
 * @param numSamples Number of samples to draw for each row slice.
 * @param seed The seed number.
 * @param normalized Whether the provided `logits` are normalized true
 *     probabilities (sum to 1). Defaults to false.
 * @return 1D array of shape `[numSamples]`, or 2D array of shape
 *     `[batchSize, numSamples]`, depending on the rank of the input.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function multinomial_(
    logits: Tensor1D|Tensor2D|TensorLike, numSamples: number, seed?: number,
    normalized = false): Tensor1D|Tensor2D {
  const $logits = convertToTensor(logits, 'logits', 'multinomial');
  const numOutcomes = $logits.size;
  const origRank = $logits.rank;
  if (numOutcomes < 2) {
    throw new Error(
        `Error in multinomial: you need at least 2 outcomes, but got ` +
        `${numOutcomes}.`);
  }
  if (origRank > 2) {
    throw new Error(`Rank of probabilities must be 1 or 2, but is ${origRank}`);
  }
  // TODO(lina128): Investigate correct seed behavior. The code seems not allow
  // setting see to 0.
  seed = seed || Math.random();

  // The kernel only accepts (and returns) rank 2 tensors.
  const logits2D: Tensor2D =
      origRank === 1 ? reshape($logits, [1, -1]) : $logits as Tensor2D;

  const inputs: MultinomialInputs = {logits: logits2D};
  const attrs: MultinomialAttrs = {numSamples, seed, normalized};

  // tslint:disable-next-line: no-unnecessary-type-assertion
  const res = ENGINE.runKernel(
                  Multinomial, inputs as {} as NamedTensorMap,
                  attrs as {} as NamedAttrMap) as Tensor2D;

  // tslint:disable-next-line:no-unnecessary-type-assertion
  return origRank === 1 ? reshape(res, [res.size]) as Tensor1D : res;
}

export const multinomial = op({multinomial_});


/**
 * Creates a `tf.Tensor` with values sampled from a uniform distribution.
 *
 * The generated values follow a uniform distribution in the range [minval,
 * maxval). The lower bound minval is included in the range, while the upper
 * bound maxval is excluded.
 *
 * ```js
 * tf.randomUniform([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param minval The lower bound on the range of random values to generate.
 *   Defaults to 0.
 * @param maxval The upper bound on the range of random values to generate.
 *   Defaults to 1.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomUniform_<R extends Rank>(
    shape: ShapeMap[R], minval = 0, maxval = 1, dtype: DataType = 'float32',
    seed?: number|string): Tensor<R> {
  const res = buffer(shape, dtype);
  const random = new UniformRandom(minval, maxval, null, seed);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = random.nextValue();
  }
  return res.toTensor();
}

export const randomUniform = op({randomUniform_});


/**
 * Creates a `tf.Tensor` with values sampled from a truncated normal
 * distribution.
 *
 * ```js
 * tf.truncatedNormal([2, 2]).print();
 * ```
 *
 * The generated values follow a normal distribution with specified mean and
 * standard deviation, except that values whose magnitude is more than 2
 * standard deviations from the mean are dropped and re-picked.
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param mean The mean of the normal distribution.
 * @param stdDev The standard deviation of the normal distribution.
 * @param dtype The data type of the output tensor.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function truncatedNormal_<R extends Rank>(
    shape: ShapeMap[R], mean = 0, stdDev = 1, dtype?: 'float32'|'int32',
    seed?: number): Tensor<R> {
  if (dtype != null && (dtype as DataType) === 'bool') {
    throw new Error(`Unsupported data type $ { dtype }`);
  }
  const randGauss =
      new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
  const res = buffer(shape, dtype);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = randGauss.nextValue();
  }
  return res.toTensor();
}

export const truncatedNormal = op({truncatedNormal_});


/**
 * Creates a `tf.Tensor` with values sampled from a gamma distribution.
 *
 * ```js
 * tf.randomGamma([2, 2], 1).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param alpha The shape parameter of the gamma distribution.
 * @param beta The inverse scale parameter of the gamma distribution. Defaults
 *     to 1.
 * @param dtype The data type of the output. Defaults to float32.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomGamma_<R extends Rank>(
    shape: ShapeMap[R], alpha: number, beta = 1,
    dtype: 'float32'|'int32' = 'float32', seed?: number): Tensor<R> {
  if (beta == null) {
    beta = 1;
  }
  if (dtype == null) {
    dtype = 'float32';
  }
  if (dtype !== 'float32' && dtype !== 'int32') {
    throw new Error(`Unsupported data type ${dtype}`);
  }
  const rgamma = new RandGamma(alpha, beta, dtype, seed);
  const res = buffer(shape, dtype);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = rgamma.nextValue();
  }
  return res.toTensor();
}

export const randomGamma = op({randomGamma_});