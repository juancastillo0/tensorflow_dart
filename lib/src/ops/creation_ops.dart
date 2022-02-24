import 'package:tensorflow_wasm/src/util_base.dart';

import '_prelude.dart';
import 'complex.dart';

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
Tensor<R> fill<R extends Rank>(
  // ShapeMap[R]
  List<int> shape,
  Object value, [
  DataType? dtype,
]) {
  final attrs = {
    'shape': shape,
    'value': value,
    'dtype': dtype ?? (value is String ? 'string' : 'float32'),
  }; // FillAttrs

  return ENGINE.runKernel(Fill, {}, attrs) as Tensor<R>;
}

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
Tensor1D linspace(double start, double stop, int numValues) {
  if (numValues <= 0) {
    throw Exception('The number of values should be positive.');
  }

  final attrs = {
    'start': start,
    'stop': stop,
    'num': numValues,
  }; // LinSpaceAttrs
  return ENGINE.runKernel(LinSpace, {}, attrs) as Tensor1D;
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
Tensor1D range(
  int start,
  int stop, {
  int? step,
  // : 'float32'|'int32'
  String dtype = 'float32',
}) {
  if (step == 0) {
    throw Exception('Cannot have a step of zero');
  }

  final attrs = {
    'start': start,
    'stop': stop,
    'step': step ?? 1,
    'dtype': dtype,
  }; // RangeAttrs

  return ENGINE.runKernel(Range, {} /* inputs */, attrs) as Tensor;
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
Tensor oneHot(
  Tensor indices,
  int depth, {
  double onValue = 1,
  double offValue = 0,
}) {
  return execOp('oneHot', () {
    if (depth < 2) {
      throw Exception('Error in oneHot: depth must be >=2, but it is ${depth}');
    }
    final $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');

    final inputs = {'indices': $indices}; // : OneHotInputs
    final attrs = {
      'depth': depth,
      'onValue': onValue,
      'offValue': offValue,
    }; // : OneHotAttrs

    return ENGINE.runKernel(OneHot, inputs, attrs) as Tensor;
  });
}

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
Tensor<R> ones<R extends Rank>(
  // : ShapeMap[R]
  List<int> shape, [
  DataType dtype = 'float32',
]) {
  if (dtype == 'complex64') {
    final real = ones<R>(shape, 'float32');
    final imag = zeros<R>(shape, 'float32');
    return complex(real, imag);
  }
  final values = makeOnesTypedArray(sizeFromShape(shape), dtype);
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
T onesLike<T extends Tensor>(T x) {
  return execOp('onesLike', () {
    final $x = convertToTensor(x, 'x', 'onesLike');

    final inputs = {'x': $x}; // : OnesLikeInputs
    return ENGINE.runKernel(OnesLike, inputs) as T;
  });
}

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
Tensor<R> zeros<R extends Rank>(
  /// : ShapeMap[R]
  List<int> shape, [
  DataType dtype = 'float32',
]) {
  if (dtype == 'complex64') {
    final real = zeros<R>(shape, 'float32');
    final imag = zeros<R>(shape, 'float32');
    return complex(real, imag);
  }
  final values = makeZerosTypedArray(sizeFromShape(shape), dtype);
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
T zerosLike<T extends Tensor>(T x) {
  return execOp('zerosLike', () {
    final $x = convertToTensor(x, 'x', 'zerosLike');
    final inputs = {'x': $x}; // ZerosLikeInputs
    return ENGINE.runKernel(ZerosLike, inputs) as T;
  });
}
