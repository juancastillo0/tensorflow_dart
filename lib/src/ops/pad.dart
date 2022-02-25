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
// import {PadV2, PadV2Attrs, PadV2Inputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {op} from './operation';

import '_prelude.dart';

/**
 * Pads a `tf.Tensor` with a given value and paddings.
 *
 * This operation implements `CONSTANT` mode. For `REFLECT` and `SYMMETRIC`,
 * refer to `tf.mirrorPad`
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that `paddings` is of given length.
 *   - `tf.pad1d`
 *   - `tf.pad2d`
 *   - `tf.pad3d`
 *   - `tf.pad4d`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.pad([[1, 2]]).print();
 * ```
 * @param x The tensor to pad.
 * @param paddings An array of length `R` (the rank of the tensor), where
 * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
 * specifying how much to pad along each dimension of the tensor.
 * @param constantValue The pad value to use. Defaults to 0.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
T pad<T extends Tensor>(
  T x,
  List<List<int>> paddings, {
  double constantValue = 0,
}) {
  return execOp('pad', () {
    final $x = convertToTensor(x, 'x', 'pad');
    if ($x.rank == 0) {
      throw Exception('pad(scalar) is not defined. Pass non-scalar to pad');
    }

    final attrs = {
      'paddings': paddings,
      'constantValue': constantValue,
    }; // : PadV2Attrs
    final inputs = {'x': $x}; // : PadV2Inputs
    return ENGINE.runKernel(PadV2, inputs, attrs) as T;
  });
}
