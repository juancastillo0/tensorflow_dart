/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// import {ENGINE} from '../../engine';
// import {customGrad} from '../../gradients';
// import {FusedDepthwiseConv2D, FusedDepthwiseConv2DAttrs, FusedDepthwiseConv2DInputs} from '../../kernel_names';
// import {NamedAttrMap} from '../../kernel_registry';
// import {Tensor, Tensor3D, Tensor4D} from '../../tensor';
// import {GradSaveFunc, NamedTensorMap} from '../../tensor_types';
// import {makeTypesMatch} from '../../tensor_util';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import * as util from '../../util';
// import {add} from '../add';
// import * as broadcast_util from '../broadcast_util';
// import * as conv_util from '../conv_util';
// import {depthwiseConv2d as unfusedDepthwiseConv2d} from '../depthwise_conv2d';
// import {depthwiseConv2dNativeBackpropFilter} from '../depthwise_conv2d_native_backprop_filter';
// import {depthwiseConv2dNativeBackpropInput} from '../depthwise_conv2d_native_backprop_input';
// import {Activation} from '../fused_types';
// import {applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse} from '../fused_util';
// import {op} from '../operation';
// import {reshape} from '../reshape';

import 'package:tensorflow_wasm/src/gradients.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

import '../_prelude.dart';
import '../broadcast_util.dart' as broadcast_util;
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import '../depthwise_conv2d.dart';
import '../depthwise_conv2d_native_backprop_filter.dart';
import '../depthwise_conv2d_native_backprop_input.dart';
import '../fused_types.dart';
import '../fused_util.dart';
import '../conv_util.dart' as conv_util;

/**
 * Computes depthwise 2D convolution, optionally fused with adding a
 * bias and applying an activation.
 *
 * Given a 4D `input` array and a `filter` array of shape
 * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
 * `inChannels` convolutional filters of depth 1, this op applies a
 * different filter to each input channel (expanding from 1 channel to
 * `channelMultiplier` channels for each), then concatenates the results
 * together. The output has `inChannels * channelMultiplier` channels.
 *
 * See
 * [https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d](
 *     https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
 * for more details.
 *
 * @param obj An object with the following properties:
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter tensor, rank 4, of shape
 *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`. If strides is a single number, then `strideHeight ==
 * strideWidth`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`).
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 * @param leakyreluAlpha Optional. Alpha to be applied as part of a `leakyrelu`
 *     activation.
 */
T fusedDepthwiseConv2d<
    T extends Tensor3D
// |Tensor4D
    >({
  required T x,
  required Tensor4D filter,
  required List<int>
      // [number, number]|number
      strides,
  required Object // 'valid'|'same'|number
      pad,
  // 'NHWC'|'NCHW'
  String dataFormat = 'NHWC',
  //  [number, number]|number?
  List<int> dilations = const [1, 1],
  // 'floor'|'round'|'ceil'?
  String? dimRoundingMode,
  Tensor? bias,
  Activation activation = Activation.linear,
  Tensor? preluActivationWeights,
  double? leakyreluAlpha,
}) {
  if (shouldFuse(ENGINE.state.gradientDepth, activation) == false) {
    var result = depthwiseConv2d(
      x,
      filter,
      strides: strides,
      pad: pad,
      dataFormat: dataFormat,
      dilations: dilations,
      dimRoundingMode: dimRoundingMode,
    );
    if (bias != null) {
      result = add(result, bias);
    }

    return applyActivation(result, activation,
        preluActivationWeights: preluActivationWeights,
        leakyreluAlpha: leakyreluAlpha) as T;
  }

  final $x = convertToTensor(x, 'x', 'depthwiseConv2d', 'float32');
  final $filter =
      convertToTensor(filter, 'filter', 'depthwiseConv2d', 'float32');

  var x4D = $x as Tensor4D;
  var reshapedTo4D = false;
  if ($x.rank == 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  util.assert_(
      x4D.rank == 4,
      () =>
          "Error in fused depthwiseConv2d: input must be rank 4, but got " +
          "rank ${x4D.rank}.");
  util.assert_(
      $filter.rank == 4,
      () =>
          "Error in fused depthwiseConv2d: filter must be rank 4, " +
          "but got rank ${$filter.rank}.");
  util.assert_(
      x4D.shape[3] == $filter.shape[2],
      () =>
          "Error in fused depthwiseConv2d: number of input channels " +
          "(${x4D.shape[3]}) must match the inChannels dimension in " +
          "filter ${$filter.shape[2]}.");
  // if (dilations == null) {
  //   dilations = [1, 1];
  // }
  util.assert_(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () =>
          'Error in fused depthwiseConv2d: Either strides or dilations must ' +
          "be 1. Got strides ${strides} and dilations '${dilations}'");
  conv_util.checkPadOnDimRoundingMode(
      'fused depthwiseConv2d', pad, dimRoundingMode);
  final convInfo = conv_util.computeConv2DInfo(
      x4D.shape, $filter.shape, strides, dilations,
      pad: pad, roundingMode: dimRoundingMode, depthwise: true /* depthwise */);

  Tensor? $bias;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused conv2d');
    $bias = makeTypesMatch($bias, $x).first;

    broadcast_util.assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
  }

  Tensor? $preluActivationWeights;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused depthwiseConv2d');
  }

  TensorList grad(Tensor4D dy, List<Tensor> saved) {
    util.assert_(
        conv_util.tupleValuesAreOne(dilations),
        () =>
            'Error in gradient of fused depthwiseConv2d: dilation rates ' +
            "greater than 1 are not yet supported. Got dilations " +
            "'${dilations}'");
    final $filter = saved[0];
    final x4D = saved[1];
    final y = saved[2];
    final bias = saved[3];

    final dyActivation = getFusedDyActivation(dy, y, activation) as Tensor4D;

    final xDer = depthwiseConv2dNativeBackpropInput(
        (x4D as Tensor4D).shape, dyActivation, $filter as Tensor4D,
        strides: strides,
        pad: pad,
        dilations: dilations,
        dimRoundingMode: dimRoundingMode);
    final filterDer = depthwiseConv2dNativeBackpropFilter(
        x4D as Tensor4D, dyActivation, ($filter as Tensor4D).shape,
        strides: strides,
        pad: pad,
        dilations: dilations,
        dimRoundingMode: dimRoundingMode);

    if (bias != null) {
      final biasDer = getFusedBiasGradient($bias!, dyActivation);
      return TensorList([xDer, filterDer, biasDer]);
    }
    return TensorList([xDer, filterDer]);
  }

  final inputs = {
    // : FusedDepthwiseConv2DInputs
    'x': x4D,
    'filter': $filter,
    if ($bias != null) 'bias': $bias,
    if ($preluActivationWeights != null)
      'preluActivationWeights': $preluActivationWeights,
  };
  final attrs = {
    // : FusedDepthwiseConv2DAttrs
    'strides': strides,
    'pad': pad,
    'dataFormat': dataFormat,
    'dilations': dilations,
    'dimRoundingMode': dimRoundingMode,
    'activation': activation,
    'leakyreluAlpha': leakyreluAlpha,
  };

  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if ($bias == null) {
    final customOp = customGrad((tensorList, save) {
      // tslint:disable-next-line: no-unnecessary-type-assertion
      var res = ENGINE.runKernel(FusedDepthwiseConv2D, inputs, attrs)
          as Tensor3D; // Tensor4D|Tensor3D

      save([...tensorList, res]);

      if (reshapedTo4D) {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]])
            as Tensor3D;
      }

      return Gradient(res, grad);
    });
    return customOp([x4D, $filter]) as T;
  } else {
    final customOpWithBias = customGrad((tensorList, save) {
      // tslint:disable-next-line: no-unnecessary-type-assertion
      var res = ENGINE.runKernel(FusedDepthwiseConv2D, inputs, attrs)
          as Tensor3D; // Tensor4D|Tensor3D

      save([...tensorList, res]);

      if (reshapedTo4D) {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]])
            as Tensor3D;
      }

      return Gradient(res, grad);
    });

    return customOpWithBias([x4D, $filter, $bias]) as T;
  }
}
