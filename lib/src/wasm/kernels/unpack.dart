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

// import {KernelConfig, KernelFunc, TensorInfo, Unpack, UnpackAttrs, UnpackInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {slice} from './Slice';

import '_prelude.dart';
import 'slice.dart';

TensorInfoList unpack({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final value = inputs['value']!;
  int axis = attrs!['axis'] as int;

  if (axis < 0) {
    axis += value.shape.length;
  }

  final numOutputs = value.shape[axis];
  final rank = value.shape.length;
  final List<int> outShape = [];

  for (int i = 0; i < rank; i++) {
    if (i != axis) {
      outShape.add(value.shape[i]);
    }
  }
  final List<TensorInfo> outs = [];
  final begin = List.filled(rank, 0);
  final size = [...value.shape];
  size[axis] = 1;
  for (int i = 0; i < outs.length; i++) {
    begin[axis] = i;
    outs.add(slice(
      inputs: {'x': value},
      attrs: {'begin': begin, 'size': size},
      backend: backend,
    ));
  }
  return TensorInfoList(
      outs.map((t) => copyTensorInfo(t, shape: outShape)).toList());
}

final unpackConfig = KernelConfigG(
  kernelName: Unpack,
  backendName: 'wasm',
  kernelFunc: unpack,
);
