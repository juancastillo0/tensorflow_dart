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

import 'package:tensorflow_wasm/src/backend_wasm.dart';
import 'package:tensorflow_wasm/src/kernel_names.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/unary_kernel.dart';

import 'concat.dart';
import 'expand_dims.dart';

// import {KernelConfig, KernelFunc, Pack, PackAttrs, PackInputs, TensorInfo, util} from '@tensorflow/tfjs-core';
// import {BackendWasm} from '../backend_wasm';

// import {concat} from './Concat';
// import {expandDims} from './ExpandDims';

TensorInfo pack({
  required List<TensorInfo> inputs,
  required BackendWasm backend,
  Map<String, Object?>? attrs,
}) {
  final axis = attrs?['axis'] as int;

  if (inputs.length == 1) {
    return expandDims(
      inputs: {'input': inputs[0]},
      backend: backend,
      attrs: {'dim': axis},
    );
  }

  final shape = inputs[0].shape;
  final dtype = inputs[0].dtype;

  inputs.forEach((t) {
    util.assertShapesMatch(shape, t.shape,
        'All tensors passed to stack must have matching shapes');
    util.assert_(dtype == t.dtype,
        () => 'All tensors passed to stack must have matching dtypes');
  });

  final List<TensorInfo> intermediateTensorInfos = [];
  final expandedTensors = inputs.map((t) {
    final expandedT = expandDims(
      inputs: {'input': t},
      backend: backend,
      attrs: {'dim': axis},
    );
    intermediateTensorInfos.add(expandedT);
    return expandedT;
  }).toList();

  final result = concat(
    inputs: expandedTensors,
    backend: backend,
    attrs: {'axis': axis},
  );

  intermediateTensorInfos.forEach((t) => backend.disposeData(t.dataId));

  return result;
}

final packConfig = KernelConfigG<BackendWasm, Map<String, Object?>>(
  kernelName: Pack,
  backendName: 'wasm',
  kernelFunc: ({required inputs, required backend, attrs}) => pack(
    inputs: List.generate(inputs.length, (index) => inputs[index.toString()]!),
    backend: backend,
    attrs: attrs,
  ),
);
