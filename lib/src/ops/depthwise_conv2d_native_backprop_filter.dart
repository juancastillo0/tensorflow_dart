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
// import {DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropFilterAttrs, DepthwiseConv2dNativeBackpropFilterInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor3D, Tensor4D} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';

// import {ExplicitPadding} from './conv_util';
// import {op} from './operation';
// import {reshape} from './reshape';

import '_prelude.dart';
import 'reshape.dart';

Tensor4D depthwiseConv2dNativeBackpropFilter<
    T extends Tensor3D
// |Tensor4D
    >(
  T x,
  T dy,
  List<int> filterShape, // : [number, number, number, number],
  {
  required List<int> strides, // : [number, number]|number,
  required Object pad, // : 'valid'|'same'|number|conv_util.ExplicitPadding,
  List<int> dilations = const [1, 1], // : [number, number]|number
  String? dimRoundingMode, // 'floor'|'round'|'ceil'
}) {
  return execOp('depthwiseConv2dNativeBackpropFilter', () {
    var x4D = x as Tensor4D;
    if (x.rank == 3) {
      x4D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]);
    }
    var dy4D = dy as Tensor4D;
    if (dy4D.rank == 3) {
      dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
    }

    final inputs = {
      'x': x4D,
      'dy': dy4D
    }; // : DepthwiseConv2dNativeBackpropFilterInputs
    final attrs = // : DepthwiseConv2dNativeBackpropFilterAttrs
        {
      'strides': strides,
      'pad': pad,
      'dimRoundingMode': dimRoundingMode,
      'dilations': dilations,
      'filterShape': filterShape,
    };

    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(DepthwiseConv2dNativeBackpropFilter, inputs, attrs)
        as Tensor4D;
  });
}
