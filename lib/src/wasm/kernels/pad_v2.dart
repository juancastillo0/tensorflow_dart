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

// import {KernelConfig, KernelFunc, PadV2, PadV2Attrs, PadV2Inputs, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {fill} from './Fill';

// import {CppDType} from './types';

import 'dart:typed_data';

import 'package:collection/collection.dart';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import '_prelude.dart';
import 'fill.dart';

late final Function(List) _wasmPadV2;
// : (xId: number, xShapeBytes: Uint8Array, xShapeLength: number, xDtype: number,
//     prePaddingsBytes: Uint8Array, postPaddingsBytes: Uint8Array,
//     constantValue: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmPadV2 = backend.wasm.cwrap(PadV2, null /* void */, [
    'number', // xId
    'array', // x.shape
    'number', // x.shape.length
    'number', // x.dtype
    'array', // pre-paddings
    'array', // post-paddings
    'number', // constantValue
    'number', // outId
  ]);
}

TensorInfo pad({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;

  final paddings = attrs!['paddings'] as List<List<int>>;
  final constantValue = attrs['constantValue'] as double;

  final outShape = paddings
      .mapIndexed(
          (i, p) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */)
      .toList();

  if (util.sizeFromShape(x.shape) == 0) {
    // Short-circuit the computation, since x doesn't have value, only
    // the shape is used to compute output shape to pad.
    return fill(
      backend: backend,
      attrs: {'shape': outShape, 'value': constantValue, 'dtype': x.dtype},
      inputs: {},
    );
  }

  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final out = backend.makeOutput(outShape, x.dtype);
  final outTensorData = backend.dataIdMap.get(out.dataId)!;
  final outId = outTensorData.id;

  final xShapeBytes = Uint8List.view(Int32List.fromList(x.shape).buffer);

  final prePaddingsFlat = paddings.map((padTuple) => padTuple[0]).toList();
  final postPaddingsFlat = paddings.map((padTuple) => padTuple[1]).toList();
  final prePaddingsBytes =
      Uint8List.view(Int32List.fromList(prePaddingsFlat).buffer);
  final postPaddingsBytes =
      Uint8List.view(Int32List.fromList(postPaddingsFlat).buffer);

  _wasmPadV2([
    xId,
    xShapeBytes,
    x.shape.length,
    CppDType.values.byName(x.dtype).index,
    prePaddingsBytes,
    postPaddingsBytes,
    constantValue,
    outId
  ]);
  return out;
}

final padV2Config = KernelConfigG(
    kernelName: PadV2, backendName: 'wasm', kernelFunc: pad, setupFunc: _setup);
