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
// import {DepthwiseConv2dNativeBackpropInput, DepthwiseConv2dNativeBackpropInputAttrs, DepthwiseConv2dNativeBackpropInputInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor3D, Tensor4D} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';

// import {ExplicitPadding} from './conv_util';
// import {op} from './operation';
// import {reshape} from './reshape';

import '../util_base.dart' as util;
import '_prelude.dart';
import 'conv_util.dart' as conv_util;
import 'reshape.dart';

T depthwiseConv2dNativeBackpropInput<
    T extends Tensor3D
// |Tensor4D
    >(
  List<int> xShape, // : [number, number, number, number]

  T dy,
  Tensor4D filter, {
  required List<int> strides, // : [number, number]|number,
  required Object pad, // : 'valid'|'same'|number|conv_util.ExplicitPadding,
  List<int> dilations = const [1, 1], // : [number, number]|number
  String? dimRoundingMode, // 'floor'|'round'|'ceil'
}) {
  return execOp('depthwiseConv2dNativeBackpropInput', () {
    var dy4D = dy as Tensor4D;
    var reshapedTo4D = false;
    if (dy.rank == 3) {
      reshapedTo4D = true;
      dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
    }

    final inputs = {
      'dy': dy4D,
      'filter': filter
    }; // : DepthwiseConv2dNativeBackpropInputInputs
    final attrs = // : DepthwiseConv2dNativeBackpropInputAttrs
        {
      'strides': strides,
      'pad': pad,
      'dimRoundingMode': dimRoundingMode,
      'dilations': dilations,
      'inputShape': xShape
    };

    final res =
        // tslint:disable-next-line: no-unnecessary-type-assertion
        ENGINE.runKernel(DepthwiseConv2dNativeBackpropInput, inputs, attrs)
            as T;

    if (reshapedTo4D) {
      return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
    }
    return res;
  });
}
