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

// import {KernelConfig, KernelFunc, Tile, TileAttrs, TileInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import 'dart:typed_data';

import '_prelude.dart';

late final Function(List) _wasmTile;
// : (
//     xId: number, xShape: Uint8Array, xShapeSize: number, newShape: Uint8Array,
//     newShapeSize: number, dtype: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmTile = backend.wasm.cwrap(Tile, null /* void */, [
    'number', // x_id
    'array', // x_shape
    'number', // x_shape.length
    'array', // new_shape
    'number', // new_shape.length
    'number' // out_id
  ]);
}

TensorInfo tile({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final reps = attrs!['reps'] as List<int>;

  final newShape = List.generate(x.shape.length, (i) => x.shape[i] * reps[i]);
  final xShapeBytes = Uint8List.view(Int32List.fromList(x.shape).buffer);
  final newShapeBytes = Uint8List.view(Int32List.fromList(newShape).buffer);

  final out = backend.makeOutput(newShape, x.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  _wasmTile([
    xId,
    xShapeBytes,
    x.shape.length,
    newShapeBytes,
    newShape.length,
    CppDType.values.byName(out.dtype).index,
    outId
  ]);
  return out;
}

final tileConfig = KernelConfigG(
  kernelName: Tile,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: tile,
);
