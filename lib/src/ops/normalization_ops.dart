import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/gradients.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

import '_prelude.dart';
import '../util_base.dart' as util;
import 'batchnorm_util.dart';

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
T softmax<T extends Tensor>(T logits, [int dim = -1]) {
  return execOp('softmax', () {
    final $logits = convertToTensor(logits, 'logits', 'softmax', 'float32');

    if (dim == -1) {
      dim = $logits.rank - 1;
    }
    if (dim != $logits.rank - 1) {
      throw Exception(
          'Softmax along a non-last dimension is not yet supported. ' +
              'Logits was rank ${$logits.rank} and dim was ${dim}');
    }

    final inputs = {'logits': $logits}; // : SoftmaxInputs
    final attrs = {'dim': dim}; // : SoftmaxAttrs

    return ENGINE.runKernel(Softmax, inputs, attrs) as T;
  });
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
T logSoftmax<T extends Tensor>(T logits, [int axis = -1]) {
  return execOp('logSoftmax', () {
    final $logits = convertToTensor(logits, 'logits', 'logSoftmax');

    if (axis == -1) {
      axis = $logits.rank - 1;
    }
    if (axis != $logits.rank - 1) {
      throw Exception(
          'Log Softmax along a non-last dimension is not yet supported. ' +
              'Logits was rank ${$logits.rank} and axis was ${axis}');
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
    final customOp = customGrad((List<Tensor> inputs, GradSaveFunc save) {
      final logits = inputs.first;
      final keepDims = true;
      final xMax = max(logits, [axis], true);
      final shifted = sub(logits, xMax);
      final value = sub(
        cast(shifted, 'float32'),
        log(sum(exp(shifted), [axis], keepDims)),
      );
      save([value]);

      gradFunc(Tensor dy, List<Tensor> saved) {
        final value = saved.first;
        final keepDims = true;
        final softmax = exp(value);
        return sub(dy, mul(sum(dy, [axis], keepDims), softmax));
      }

      return Gradient(value, gradFunc);
    });

    return customOp([$logits]) as T;

    // TODO Use Engine.runKernel when CPU/WebGL/WASM backends implement this.
    // const inputs: LogSoftmaxInputs = {logits: $logits};
    // const attrs: LogSoftmaxAttrs = {axis};
    // return ENGINE.runKernel(
    //            LogSoftmax, inputs as {} as NamedTensorMap,
    //            attrs as {} as NamedAttrMap);
  });
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
T localResponseNormalization<
    T extends Tensor
// Tensor3D|Tensor4D
    >(
  T x, {
  int depthRadius = 5,
  double bias = 1,
  double alpha = 1,
  double beta = 0.5,
}) {
  return execOp('localResponseNormalization', () {
    final $x = convertToTensor(x, 'x', 'localResponseNormalization');
    util.assert_(
        $x.rank == 4 || $x.rank == 3,
        () =>
            'Error in localResponseNormalization: x must be rank 3 or 4 but got rank ${$x.rank}.');
    util.assert_(
        depthRadius is int,
        () =>
            'Error in localResponseNormalization: depthRadius must be an ' +
            'integer but got depthRadius ${depthRadius}.');

    Tensor x4D = $x; // as Tensor4D;
    var reshapedTo4D = false;
    if ($x.rank == 3) {
      reshapedTo4D = true;
      x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }

    final inputs = {'x': x4D}; // : LRNInputs

    final attrs = {
      'depthRadius': depthRadius,
      'bias': bias,
      'alpha': alpha,
      'beta': beta,
    }; // : LRNAttrs

    // tslint:disable-next-line: no-unnecessary-type-assertion
    final res = ENGINE.runKernel(LRN, inputs, attrs) as T;

    if (reshapedTo4D) {
      return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
    } else {
      return res;
    }
  });
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
Tensor<R> batchNorm<R extends Rank>(
  Tensor<R> x,
  // : Tensor<R>|Tensor1D|TensorLike
  Tensor<R> mean,
  Tensor<R> variance, {
  Tensor<R>? offset,
  Tensor<R>? scale,
  double varianceEpsilon = 0.001,
}) {
  return execOp('batchNorm', () {
    final $x = convertToTensor(x, 'x', 'batchNorm');
    final $mean = convertToTensor(mean, 'mean', 'batchNorm');
    final $variance = convertToTensor(variance, 'variance', 'batchNorm');
    Tensor<R>? $scale; //: Tensor<R>|Tensor1D;
    if (scale != null) {
      $scale = convertToTensor(scale, 'scale', 'batchNorm');
    }
    Tensor<R>? $offset; //: Tensor<R>|Tensor1D;
    if (offset != null) {
      $offset = convertToTensor(offset, 'offset', 'batchNorm');
    }

    util.assert_(
        $mean.rank == $variance.rank,
        () =>
            'Batch normalization gradient requires mean and variance to have ' +
            'equal ranks.');
    util.assert_(
        $offset == null || $mean.rank == $offset.rank,
        () =>
            'Batch normalization gradient requires mean and offset to have ' +
            'equal ranks.');
    util.assert_(
        $scale == null || $mean.rank == $scale.rank,
        () =>
            'Batch normalization gradient requires mean and scale to have ' +
            'equal ranks.');

    final x4D = xAs4D($x);

    final inputs = {
      //  FusedBatchNormInputs
      'x': x4D,
      if ($scale != null) 'scale': $scale,
      if ($offset != null) 'offset': $offset,
      'mean': $mean,
      'variance': $variance
    };

    final attrs = {'varianceEpsilon': varianceEpsilon}; // : FusedBatchNormAttrs

    // tslint:disable-next-line: no-unnecessary-type-assertion
    final res = ENGINE.runKernel(FusedBatchNorm, inputs, attrs) as Tensor<R>;

    return reshape(res, $x.shape);
  });
}
