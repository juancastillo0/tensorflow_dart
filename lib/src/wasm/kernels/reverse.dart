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

// import {KernelConfig, KernelFunc, Reverse, ReverseAttrs, ReverseInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {identity} from './Identity';
// import {reshape} from './Reshape';

import 'dart:typed_data';

import '_prelude.dart';
import '../../util_base.dart' as util;
import 'identity.dart';
import 'reshape.dart';

late final Function(List) _wasmReverse;
// : (
//     xId: number, axes: Uint8Array, axesLength: number, outShape: Uint8Array,
//     outShapeLength: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmReverse = backend.wasm.cwrap(Reverse, null, [
    'number', // x_id
    'array', // axes
    'number', // axes_length
    'array', // out_shape
    'number', // out_shape_length
    'number' // out_id
  ]);
}

TensorInfo reverse({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final dims = attrs!['dims'] as List<int>;

  final axes = util.parseAxisParam(dims, x.shape);

  if (x.shape.length == 0) {
    return identity(inputs: {'x': x}, backend: backend);
  }

  final out = backend.makeOutput(x.shape, x.dtype);
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final axesBytes = Uint8List.view(Int32List.fromList(axes).buffer);
  final outShapeBytes = Uint8List.view(Int32List.fromList(x.shape).buffer);

  _wasmReverse(
      [xId, axesBytes, axes.length, outShapeBytes, x.shape.length, outId]);

  final reshaped = reshape(
    inputs: {'x': out},
    attrs: {'shape': x.shape},
    backend: backend,
  );

  backend.disposeData(out.dataId);
  return reshaped;
}

final reverseConfig = KernelConfigG(
  kernelName: Reverse,
  backendName: 'wasm',
  kernelFunc: reverse,
  setupFunc: _setup,
);
