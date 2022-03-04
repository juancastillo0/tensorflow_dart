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

// import {KernelConfig, KernelFunc, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {reshape} from './Reshape';
// import {slice} from './Slice';
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;
import '../../util_base.dart' as util;
import 'dart:typed_data';

import '_prelude.dart';
import 'reshape.dart';
import 'slice.dart';

late final Function(List) _wasmStridedSlice;
// : (
//     xId: number, xStridesBytes: Uint8Array, xRank: number,
//     beginBytes: Uint8Array, endBytes: Uint8Array, stridesBytes: Uint8Array,
//     outShapeBytes: Uint8Array, outStridesBytes: Uint8Array,
//     outShapeLength: number, outId: number) => void;

void setup(BackendWasm backend) {
  _wasmStridedSlice = backend.wasm.cwrap(StridedSlice, null /*void*/, [
    'number', // xId
    'array', // xStrides
    'number', // xRank
    'array', // beginBytes
    'array', // endBytes
    'array', // stridesBytes
    'array', // outShapeBytes
    'array', // outStridesBytes
    'number', // outShapeLength
    'number', // outId
  ]);
}

TensorInfo stridedSlice({
  required NamedTensorInfoMap inputs,
  StridedSliceAttrs? attrs,
  required BackendWasm backend,
}) {
  final x = inputs['x']!;

  final begin = attrs!.begin;
  final end = attrs.end;
  final strides = attrs.strides;
  final beginMask = attrs.beginMask;
  final endMask = attrs.endMask;
  final ellipsisMask = attrs.ellipsisMask;
  final newAxisMask = attrs.newAxisMask;
  final shrinkAxisMask = attrs.shrinkAxisMask;

  final info = backend_util.sliceInfo(x.shape, begin, end, strides, beginMask,
      endMask, ellipsisMask, newAxisMask, shrinkAxisMask);

  final finalShapeSparse = info.finalShapeSparse;
  final finalShape = info.finalShape;
  final isIdentity = info.isIdentity;
  final sliceDim0 = info.sliceDim0;
  final isSimpleSlice = info.isSimpleSlice;
  final $begin = info.begin;
  final $end = info.end;
  final $strides = info.strides;

  final TensorInfo result;

  if (isIdentity) {
    // Optimization #1, slice is a no-op plus reshape
    result = reshape(
        inputs: {'x': x}, backend: backend, attrs: {'shape': finalShape});
  } else if (sliceDim0 || isSimpleSlice) {
    // Optimization #2, slice is memory contiguous (only occurs in dim 0)
    util.assert_(x.shape.length >= 1,
        () => 'Input must have rank at least 1, got: ${x.shape.length}');

    final size = backend_util.computeSliceOutShape($begin, $end, $strides);
    // To tolerate begin[0] > end[0] (a 0-output slice), we min(begin, end).
    final sliced = slice(
      inputs: {'x': x},
      backend: backend,
      attrs: {'begin': $begin, 'size': size},
    );
    result = reshape(
      inputs: {'x': sliced},
      backend: backend,
      attrs: {'shape': finalShape},
    );
    backend.disposeData(sliced.dataId);
  } else {
    final out = backend.makeOutput(finalShapeSparse, 'float32');

    final xId = backend.dataIdMap.get(x.dataId)!.id;
    final xStridesBytes =
        Uint8List.view(Int32List.fromList(util.computeStrides(x.shape)).buffer);
    final beginBytes = Uint8List.view(Int32List.fromList($begin).buffer);
    final endBytes = Uint8List.view(Int32List.fromList($end).buffer);
    final stridesBytes = Uint8List.view(Int32List.fromList($strides).buffer);

    final outputShapeBytes =
        Uint8List.view(Int32List.fromList(finalShapeSparse).buffer);
    final outStridesBytes = Uint8List.view(
        Int32List.fromList(util.computeStrides(finalShapeSparse)).buffer);
    final outId = backend.dataIdMap.get(out.dataId)!.id;

    _wasmStridedSlice([
      xId,
      xStridesBytes,
      x.shape.length,
      beginBytes,
      endBytes,
      stridesBytes,
      outputShapeBytes,
      outStridesBytes,
      finalShapeSparse.length,
      outId
    ]);

    result = reshape(
      inputs: {'x': out},
      backend: backend,
      attrs: {'shape': finalShape},
    );

    backend.disposeData(out.dataId);
  }

  return result;
}

final stridedSliceConfig = KernelConfigG(
  kernelName: StridedSlice,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: stridedSlice,
);
