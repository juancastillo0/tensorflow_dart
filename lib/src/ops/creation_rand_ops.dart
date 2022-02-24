import 'dart:math' as math;
import '_prelude.dart';
import 'buffer.dart';
import 'rand_util.dart';
import 'reshape.dart';

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
Tensor1D multinomial(
  // Tensor1D|Tensor2D
  Tensor logits,
  int numSamples, {
  int? seed,
  bool normalized = false,
}) {
  return execOp('multinomial', () {
    final $logits = convertToTensor(logits, 'logits', 'multinomial');
    final numOutcomes = $logits.size;
    final origRank = $logits.rank;
    if (numOutcomes < 2) {
      throw Exception(
          'Error in multinomial: you need at least 2 outcomes, but got ' +
              '${numOutcomes}.');
    }
    if (origRank > 2) {
      throw Exception(
          'Rank of probabilities must be 1 or 2, but is ${origRank}');
    }
    // TODO(lina128): Investigate correct seed behavior. The code seems not allow
    // setting see to 0.
    seed = seed ?? math.Random().nextInt((1e10).toInt());

    // The kernel only accepts (and returns) rank 2 tensors.
    final Tensor2D logits2D =
        origRank == 1 ? reshape($logits, [1, -1]) : $logits as Tensor2D;

    final inputs = {'logits': logits2D}; // : MultinomialInputs
    final attrs = {
      'numSamples': numSamples,
      'seed': seed,
      'normalized': normalized,
    }; // : MultinomialAttrs

    // tslint:disable-next-line: no-unnecessary-type-assertion
    final res = ENGINE.runKernel(Multinomial, inputs, attrs) as Tensor2D;

    // tslint:disable-next-line:no-unnecessary-type-assertion
    return origRank == 1 ? reshape(res, [res.size]) as Tensor1D : res;
  });
}

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
 * @param min The lower bound on the range of random values to generate.
 *   Defaults to 0.
 * @param max The upper bound on the range of random values to generate.
 *   Defaults to 1.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
Tensor<R> randomUniform<R extends Rank>(
  // : ShapeMap[R]
  Shape shape, {
  double min = 0,
  double max = 1,
  DataType dtype = 'float32',
  int? seed,
}) {
  return execOp('randomUniform', () {
    final res = buffer(shape, dtype, null);
    final random = UniformRandom(min: min, max: max, seed: seed);
    for (int i = 0; i < res.values.length; i++) {
      res.values[i] = random.nextValue();
    }
    return res.toTensor() as Tensor<R>;
  });
}

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
Tensor<R> truncatedNormal<R extends Rank>(
  // : ShapeMap[R]
  Shape shape, {
  double mean = 0,
  double stdDev = 1,
  // : 'float32'|'int32'
  DataType dtype = 'float32',
  int? seed,
}) {
  return execOp('randomUniform', () {
    if (dtype != null && (dtype as DataType) == 'bool') {
      throw Exception('Unsupported data type ${dtype}');
    }
    final randGauss = MPRandGauss(
        mean: mean,
        stdDev: stdDev,
        truncated: true /* truncated */,
        seed: seed);
    final res = buffer(shape, dtype, null);
    final isInt = dtype == 'int32';
    for (int i = 0; i < res.values.length; i++) {
      final v = randGauss.nextValue();
      res.values[i] = isInt ? v.round() : v;
    }
    return res.toTensor() as Tensor<R>;
  });
}

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
Tensor<R> randomGamma<R extends Rank>(
  // : ShapeMap[R]
  Shape shape,
  double alpha, {
  double beta = 1,
  // : 'float32'|'int32'
  DataType dtype = 'float32',
  int? seed,
}) {
  return execOp('randomGamma', () {
    if (dtype != 'float32' && dtype != 'int32') {
      throw Exception('Unsupported data type ${dtype}');
    }
    final rgamma = RandGamma(alpha: alpha, beta: beta, seed: seed);
    final res = buffer(shape, dtype, null);
    final isInt = dtype == 'int32';
    for (int i = 0; i < res.values.length; i++) {
      final v = rgamma.nextValue();
      res.values[i] = isInt ? v.round() : v;
    }
    return res.toTensor() as Tensor<R>;
  });
}
