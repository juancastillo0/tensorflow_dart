import 'package:tensorflow_wasm/src/ops/tensor_ops_util.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/tensor_util_env.dart';

// import {Tensor} from '../tensor';
// import {inferShape} from '../tensor_util_env';
// import {TensorLike} from '../types';
// import {DataType, Rank, ShapeMap} from '../types';

// import {makeTensor} from './tensor_ops_util';

/**
 * Creates a `tf.Tensor` with the provided values, shape and dtype.
 *
 * ```js
 * // Pass an array of values to create a vector.
 * tf.tensor([1, 2, 3, 4]).print();
 * ```
 *
 * ```js
 * // Pass a nested array of values to make a matrix or a higher
 * // dimensional tensor.
 * tf.tensor([[1, 2], [3, 4]]).print();
 * ```
 *
 * ```js
 * // Pass a flat array and specify a shape yourself.
 * tf.tensor([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`. If the values are strings,
 *     they will be encoded as utf-8 and kept as `Uint8Array[]`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
Tensor<R> tensor<R extends Rank>(
  TensorLike values, [
  List<int>? shape,
  DataType? dtype,
]) {
  final inferredShape = inferShape(values, dtype);
  return makeTensor(values, shape, inferredShape, dtype) as Tensor<R>;
}

/**
 * Creates rank-1 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor1d` as it makes the code more readable.
 *
 * ```js
 * tf.tensor1d([1, 2, 3]).print();
 * ```
 *
 * @param values The values of the tensor. Can be array of numbers,
 *     or a `TypedArray`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
Tensor1D tensor1d(Object values, [DataType? dtype]) {
  // assertNonNull(values);
  final inferredShape = inferShape(values, dtype);
  if (inferredShape.length != 1) {
    throw Exception('tensor1d() requires values to be a flat/TypedArray');
  }
  final shape = null;
  return makeTensor(values, shape, inferredShape, dtype) as Tensor1D;
}
