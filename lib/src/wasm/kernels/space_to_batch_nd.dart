/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

// import {backend_util, KernelConfig, KernelFunc, ReshapeAttrs, ReshapeInputs, SpaceToBatchND, SpaceToBatchNDAttrs, SpaceToBatchNDInputs, TensorInfo, TransposeAttrs, TransposeInputs, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {padV2Config} from './PadV2';
// import {reshape} from './Reshape';
// import {transpose} from './Transpose';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import '_prelude.dart';
import 'pad_v2.dart';
import 'reshape.dart';
import 'transpose.dart';

TensorInfo spaceToBatchND({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;

  final paddings = attrs!['paddings'] as List<List<int>>;
  final blockShape = attrs['blockShape'] as List<int>;

  final prod = util.sizeFromShape(blockShape);

  final completePaddings = [
    [0, 0]
  ];
  completePaddings.addAll(paddings);

  for (int i = 1 + blockShape.length; i < x.shape.length; ++i) {
    completePaddings.add([0, 0]);
  }

  final paddedX = padV2Config.kernelFunc(
      inputs: {'x': x},
      backend: backend,
      attrs: {'paddings': completePaddings, 'finalantValue': 0}) as TensorInfo;

  final reshapedPaddedShape =
      backend_util.getReshaped(paddedX.shape, blockShape, prod, false);

  final permutedReshapedPaddedPermutation = backend_util.getPermuted(
      reshapedPaddedShape.length, blockShape.length, false);

  final flattenShape =
      backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);

  final reshapeInputs = {'x': paddedX}; // : ReshapeInputs
  final reshapeAttrs = {'shape': reshapedPaddedShape}; // : ReshapeAttrs
  final paddedXReshaped =
      reshape(inputs: reshapeInputs, backend: backend, attrs: reshapeAttrs);

  final transposeInputs = {'x': paddedXReshaped}; // : TransposeInputs
  final transposeAttrs = {
    'perm': permutedReshapedPaddedPermutation
  }; // : TransposeAttrs
  final paddedXT = transpose(
      inputs: transposeInputs, backend: backend, attrs: transposeAttrs);

  final resultReshapeInputs = {'x': paddedXT}; // : ReshapeInputs
  final resultReshapeAttrs = {'shape': flattenShape}; // : ReshapeAttrs
  final result = reshape(
      inputs: resultReshapeInputs, backend: backend, attrs: resultReshapeAttrs);

  backend.disposeData(paddedX.dataId);
  backend.disposeData(paddedXReshaped.dataId);
  backend.disposeData(paddedXT.dataId);

  return result;
}

final spaceToBatchNDConfig = KernelConfigG(
  kernelName: SpaceToBatchND,
  backendName: 'wasm',
  kernelFunc: spaceToBatchND,
);
