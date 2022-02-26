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
// import {Tensor3D, Tensor4D} from '../tensor';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {conv2DBackpropInput} from './conv2d_backprop_input';
// import {ExplicitPadding} from './conv_util';
// import {op} from './operation';

import '../util_base.dart' as util;
import '_prelude.dart';
import 'conv2d_backprop_input.dart';
import 'conv_util.dart' as conv_util;
import 'reshape.dart';

/**
 * Computes the transposed 2D convolution of an image, also known as a
 * deconvolution.
 *
 * @param x The input image, of rank 4 or rank 3, of shape
 *   `[batch, height, width, inDepth]`. If rank 3, batch of 1 is assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, outDepth, inDepth]`.
 *     `inDepth` must match `inDepth` in `x`.
 * @param outputShape Output shape, of rank 4 or rank 3:
 *     `[batch, height, width, outDepth]`. If rank 3, batch of 1 is assumed.
 * @param strides The strides of the original convolution:
 *     `[strideHeight, strideWidth]`.
 * @param pad  The type of padding algorithm used in the non-transpose version
 *    of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
T conv2dTranspose<
    T extends Tensor3D
//|Tensor4D
    >(
  T x,
  Tensor4D filter, {
  required List<int>
      outputShape, // : [number, number, number, number]|[number, number, number]
  required List<int> strides, // : [number, number]|number,
  required Object pad, // : 'valid'|'same'|number|conv_util.ExplicitPadding,
  String? dimRoundingMode, // 'floor'|'round'|'ceil'
}) {
  return execOp('conv2dTranspose', () {
    final $x = convertToTensor(x, 'x', 'conv2dTranspose');
    final $filter = convertToTensor(filter, 'filter', 'conv2dTranspose');

    return conv2DBackpropInput(
      outputShape,
      $x,
      $filter,
      strides: strides,
      pad: pad,
      dataFormat: 'NHWC',
      dimRoundingMode: dimRoundingMode,
    );
  });
}
