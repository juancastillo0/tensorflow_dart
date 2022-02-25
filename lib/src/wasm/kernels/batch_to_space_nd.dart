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

// import {backend_util, BatchToSpaceND, BatchToSpaceNDAttrs, BatchToSpaceNDInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {reshape} from './Reshape';
// import {slice} from './Slice';
// import {transpose} from './Transpose';

import '_prelude.dart';

import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'reshape.dart';
import 'slice.dart';
import 'transpose.dart';

TensorInfo batchToSpaceND({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;

  final crops = attrs!['crops'] as List<List<int>>;
  final blockShape = attrs['blockShape'] as List<int>;

  final prod = blockShape.reduce((a, b) => a * b);

  final reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
  final permuted = backend_util.getPermuted(reshaped.length, blockShape.length);
  final reshapedPermuted =
      backend_util.getReshapedPermuted(x.shape, blockShape, prod);
  final sliceBeginCoords =
      backend_util.getSliceBeginCoords(crops, blockShape.length);
  final sliceSize =
      backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

  final xReshaped =
      reshape(inputs: {'x': x}, backend: backend, attrs: {'shape': reshaped});
  final xTransposed = transpose(
      inputs: {'x': xReshaped}, backend: backend, attrs: {'perm': permuted});
  final xTransposedReshaped = reshape(
      inputs: {'x': xTransposed},
      backend: backend,
      attrs: {'shape': reshapedPermuted});
  final result = slice(
      inputs: {'x': xTransposedReshaped},
      backend: backend,
      attrs: {'begin': sliceBeginCoords, 'size': sliceSize});

  backend.disposeData(xReshaped.dataId);
  backend.disposeData(xTransposed.dataId);
  backend.disposeData(xReshaped.dataId);

  return result;
}

final batchToSpaceNDConfig = KernelConfigG(
  kernelName: BatchToSpaceND,
  backendName: 'wasm',
  kernelFunc: batchToSpaceND,
);
