/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import {ENGINE} from '../engine';
// import {Reshape, ReshapeAttrs, ReshapeInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {Rank, ShapeMap, TensorLike} from '../types';

// import {op} from './operation';

import 'package:tensorflow_wasm/src/util_base.dart' show squeezeShape;

import '_prelude.dart';

/**
 * Reshapes a `tf.Tensor` to a given shape.
 *
 * Given an input tensor, returns a new tensor with the same values as the
 * input tensor with shape `shape`.
 *
 * If one component of shape is the special value -1, the size of that
 * dimension is computed so that the total size remains constant. In
 * particular, a shape of [-1] flattens into 1-D. At most one component of
 * shape can be -1.
 *
 * If shape is 1-D or higher, then the operation returns a tensor with shape
 * shape filled with the values of tensor. In this case, the number of
 * elements implied by shape must be the same as the number of elements in
 * tensor.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.reshape([2, 2]).print();
 * ```
 *
 * @param x The input tensor to be reshaped.
 * @param shape An array of integers defining the output tensor shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
Tensor<R> reshape<R extends Rank>(
  Tensor x,
  // ShapeMap[R]
  List<int> shape,
) {
  return execOp('reshape', () {
    final $x = convertToTensor(x, 'x', 'reshape', 'string_or_numeric');

    final inputs = {'x': $x}; // ReshapeInputs
    final attrs = {'shape': shape}; // ReshapeAttrs
    return ENGINE.runKernel(Reshape, inputs, attrs) as Tensor<R>;
  });
}

/**
 * Removes dimensions of size 1 from the shape of a `tf.Tensor`.
 *
 * ```js
 * const x = tf.tensor([1, 2, 3, 4], [1, 1, 4]);
 * x.squeeze().print();
 * ```
 *
 * @param x The input tensor to be squeezed.
 * @param axis An optional list of numbers. If specified, only
 *     squeezes the dimensions listed. The dimension index starts at 0. It
 * is an error to squeeze a dimension that is not 1.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
T squeeze<T extends Tensor>(Tensor x, [List<int>? axis]) {
  return execOp('squeeze', () {
    final $x = convertToTensor(x, 'x', 'squeeze');
    return reshape($x, squeezeShape($x.shape, axis).newShape) as T;
  });
}

/**
 * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
 * into the tensor's shape.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const axis = 1;
 * x.expandDims(axis).print();
 * ```
 *
 * @param x The input tensor whose dimensions to be expanded.
 * @param axis The dimension index at which to insert shape of `1`. Defaults
 *     to 0 (the first dimension).
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function expandDims_<T extends Tensor>(x: Tensor|TensorLike, axis = 0): T {
  const $x = convertToTensor(x, 'x', 'expandDims', 'string_or_numeric');

  util.assert(axis <= $x.rank, () => 'Axis must be <= rank of the tensor');

  const inputs: ExpandDimsInputs = {input: $x};
  const attrs: ExpandDimsAttrs = {dim: axis};

  return ENGINE.runKernel(
      ExpandDims, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const expandDims = op({expandDims_});
