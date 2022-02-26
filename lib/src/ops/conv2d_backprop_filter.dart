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
// import {Conv2DBackpropFilter, Conv2DBackpropFilterAttrs, Conv2DBackpropFilterInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor3D, Tensor4D} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import * as util from '../util';

// import * as conv_util from './conv_util';
// import {op} from './operation';
// import {reshape} from './reshape';

import '../util_base.dart' as util;
import '_prelude.dart';
import 'conv_util.dart' as conv_util;
import 'reshape.dart';

/**
 * Computes the derivative of the filter of a 2D convolution.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
 * @param dy The dy image, of rank 4 or rank 3, of shape
 *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
 * @param filterShape The shape of the filter, length 4,
 *     [filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
Tensor4D conv2DBackpropFilter<
    T extends Tensor3D
//|Tensor4D
    >(
  T x,
  T dy,
  List<int> filterShape, {
  required List<int> strides, // : [number, number]|number,
  required Object pad, // : 'valid'|'same'|number|conv_util.ExplicitPadding,
  String dataFormat = 'NHWC', // : 'NHWC'|'NCHW'
  String? dimRoundingMode, // 'floor'|'round'|'ceil'
}) {
  return execOp('conv2DBackpropFilter', () {
    var x4D = x as Tensor4D;
    if (x.rank == 3) {
      x4D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]);
    }
    var dy4D = dy as Tensor4D;
    if (dy4D.rank == 3) {
      dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
    }
    util.assert_(
        x4D.rank == 4,
        () =>
            'Error in conv2dDerFilter: input must be rank 4, but got shape ' +
            '${x4D.shape}.');
    util.assert_(
        dy4D.rank == 4,
        () =>
            'Error in conv2dDerFilter: dy must be rank 4, but got shape ' +
            '${dy4D.shape}.');
    util.assert_(
        filterShape.length == 4,
        () =>
            'Error in conv2dDerFilter: filterShape must be length 4, but got ' +
            '${filterShape}.');
    final inDepth = dataFormat == 'NHWC' ? x4D.shape[3] : x4D.shape[1];
    final outDepth = dataFormat == 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
    util.assert_(
        inDepth == filterShape[2],
        () =>
            'Error in conv2dDerFilter: depth of input ${inDepth}) must ' +
            'match input depth in filter (${filterShape[2]}.');
    util.assert_(
        outDepth == filterShape[3],
        () =>
            'Error in conv2dDerFilter: depth of dy (${outDepth}) must ' +
            'match output depth for filter (${filterShape[3]}).');
    conv_util.checkPadOnDimRoundingMode(
        'conv2dDerFilter', pad, dimRoundingMode);
    final inputs = {'x': x4D, 'dy': dy4D}; // Conv2DBackpropFilterInputs
    final attrs = // Conv2DBackpropFilterAttrs
        {
      'strides': strides,
      'pad': pad,
      'dataFormat': dataFormat,
      'dimRoundingMode': dimRoundingMode,
      'filterShape': filterShape,
    };

    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(Conv2DBackpropFilter, inputs, attrs) as Tensor4D;
  });
}
