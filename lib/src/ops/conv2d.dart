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
// import {Conv2D, Conv2DAttrs, Conv2DInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor3D, Tensor4D} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';
// import * as util from '../util';

// import * as conv_util from './conv_util';
// import {op} from './operation';
// import {reshape} from './reshape';

import '../util_base.dart' as util;
import '_prelude.dart';
import 'conv_util.dart' as conv_util;
import 'reshape.dart';

/**
 * Computes a 2D convolution over the input x.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
T conv2d<
    T extends Tensor3D
// |Tensor4D
    >(
  T x,
  Tensor4D filter, {
  required List<int> strides, // : [number, number]|number,
  required Object pad, // : 'valid'|'same'|number|conv_util.ExplicitPadding,
  String dataFormat = 'NHWC', // : 'NHWC'|'NCHW'
  List<int> dilations = const [1, 1], // : [number, number]|number
  String? dimRoundingMode, // 'floor'|'round'|'ceil'
}) {
  return execOp('conv2d', () {
    final $x = convertToTensor(x, 'x', 'conv2d', 'float32');
    final $filter = convertToTensor(filter, 'filter', 'conv2d', 'float32');

    var x4D = $x as Tensor4D;
    var reshapedTo4D = false;

    if ($x.rank == 3) {
      reshapedTo4D = true;
      x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }

    util.assert_(
        x4D.rank == 4,
        () =>
            "Error in conv2d: input must be rank 4, but got rank ${x4D.rank}.");
    util.assert_(
        $filter.rank == 4,
        () =>
            "Error in conv2d: filter must be rank 4, but got rank " +
            "${$filter.rank}.");
    conv_util.checkPadOnDimRoundingMode('conv2d', pad, dimRoundingMode);
    final inDepth = dataFormat == 'NHWC' ? x4D.shape[3] : x4D.shape[1];
    util.assert_(
        inDepth == $filter.shape[2],
        () =>
            "Error in conv2d: depth of input (${inDepth}) must match " +
            "input depth for filter ${$filter.shape[2]}.");
    util.assert_(
        conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
        () =>
            'Error in conv2D: Either strides or dilations must be 1. ' +
            "Got strides ${strides} and dilations '${dilations}'");

    final inputs = {'x': x4D, 'filter': $filter}; // Conv2DInputs
    final attrs = {
      'strides': strides,
      'pad': pad,
      'dataFormat': dataFormat,
      'dilations': dilations,
      'dimRoundingMode': dimRoundingMode,
    }; // Conv2DAttrs

    // tslint:disable-next-line: no-unnecessary-type-assertion
    final res = ENGINE.runKernel(Conv2D, inputs, attrs) as T;

    if (reshapedTo4D) {
      return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
    }
    return res;
  });
}
