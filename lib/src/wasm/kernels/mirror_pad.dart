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

// import {KernelConfig, KernelFunc, MirrorPad, MirrorPadAttrs, MirrorPadInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

import 'dart:typed_data';

import 'package:collection/collection.dart';
import '_prelude.dart';

// Must match enum in MirrorPad.cc
enum MirrorPaddingMode { reflect, symmetric }

late final Function(List) _wasmMirrorPad;
// : ( xId: number, xShapeBytes: Uint8Array, xShapeLength: number, xDtype: number,
//     prePaddingsBytes: Uint8Array, postPaddingsBytes: Uint8Array, mode: number,
//     outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmMirrorPad = backend.wasm.cwrap(MirrorPad, null /* void */, [
    'number', // xId
    'array', // x.shape
    'number', // x.shape.length
    'number', // x.dtype
    'array', // pre-paddings
    'array', // post-paddings
    'number', // mode
    'number', // outId
  ]);
}

TensorInfo mirrorPad({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;

  final paddings = attrs!['paddings'] as List<List<int>>;
  final mode = attrs['mode'] as String;

  final outShape = paddings
      .mapIndexed(
          (i, p) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */)
      .toList();
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final out = backend.makeOutput(outShape, x.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;
  final xShapeBytes = Uint8List.view(Int32List.fromList(x.shape).buffer);

  final prePaddingsFlat = paddings.map((padTuple) => padTuple[0]).toList();
  final postPaddingsFlat = paddings.map((padTuple) => padTuple[1]).toList();
  final prePaddingsBytes =
      Uint8List.view(Int32List.fromList(prePaddingsFlat).buffer);
  final postPaddingsBytes =
      Uint8List.view(Int32List.fromList(postPaddingsFlat).buffer);

  _wasmMirrorPad([
    xId,
    xShapeBytes,
    x.shape.length,
    CppDType.values.byName(x.dtype).index,
    prePaddingsBytes,
    postPaddingsBytes,
    MirrorPaddingMode.values.byName(mode).index,
    outId
  ]);
  return out;
}

final mirrorPadConfig = KernelConfigG(
    kernelName: MirrorPad,
    backendName: 'wasm',
    kernelFunc: mirrorPad,
    setupFunc: _setup);
