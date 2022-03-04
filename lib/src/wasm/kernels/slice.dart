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

// import {backend_util, KernelConfig, KernelFunc, Slice, slice_util, SliceAttrs, SliceInputs, TypedArray, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {sliceImplCPU} from '../kernel_utils/shared';

import 'dart:typed_data';

import 'package:collection/collection.dart';

import '../kernel_utils/shared.dart' show sliceImplCPU;
import '_prelude.dart';
import 'package:tensorflow_wasm/slice_util.dart' as slice_util;
import 'package:tensorflow_wasm/src/util_base.dart' as util;

TensorInfo slice({
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
  required BackendWasm backend,
}) {
  final x = inputs['x']!;
  final begin = parseAxis(attrs!['begin']!);
  final size = parseAxis(attrs['size']!);

  final _p = slice_util.parseSliceParams(x, begin, size);
  final begin_ = _p[0];
  final size_ = _p[1];

  final isContinous = slice_util.isSliceContinous(x.shape, begin_, size_);
  final xVals = backend.readSync(x.dataId);
  final out = backend.makeOutput(size_, x.dtype);
  final xStrides = util.computeStrides(x.shape);
  final outData = backend.dataIdMap.get(out.dataId)!;

  if (isContinous) {
    final flatOffset = slice_util.computeFlatOffset(begin_, xStrides);

    if (x.dtype == 'string') {
      outData.stringBytes = (xVals as List<Uint8List>)
          .slice(flatOffset, flatOffset + util.sizeFromShape(size_));
    } else {
      final outVals = backend.typedArrayFromHeap(out) as List;
      List.copyRange(
        outVals,
        0,
        xVals,
        flatOffset,
        flatOffset + util.sizeFromShape(size_),
      );
    }

    return out;
  }

  if (x.dtype == 'string') {
    final res = sliceImplCPU(xVals, begin_, size_, x.shape, x.dtype);
    outData.stringBytes = res as List<Uint8List>;
    return out;
  }

  final outVals = backend.typedArrayFromHeap(out) as List;
  final rank = x.shape.length;
  if (rank == 2) {
    slice2d(xVals, xStrides[0], outVals, begin_, size_);
  } else if (rank == 3) {
    slice3d(xVals, xStrides[0], xStrides[1], outVals, begin_, size_);
  } else if (rank == 4) {
    slice4d(
        xVals, xStrides[0], xStrides[1], xStrides[2], outVals, begin_, size_);
  } else {
    final res = sliceImplCPU(xVals, begin_, size_, x.shape, x.dtype);
    outVals.setAll(0, res);
  }

  return out;
}

void slice2d(
  List xVals,
  int xStride,
  List outVals,
  // : [number, number]
  List<int> begin,
  // : [number, number]
  List<int> size,
) {
  int outOffset = 0;
  final beginI = begin[0];
  final beginJ = begin[1];
  final endI = beginI + size[0];
  for (int i = beginI; i < endI; i++) {
    final xOffset = i * xStride + beginJ;
    outVals.setAll(outOffset, xVals.slice(xOffset, xOffset + size[1]));
    outOffset += size[1];
  }
}

void slice3d(
  // : backend_util.TypedArray
  List xVals,
  int xStride1,
  int xStride2,
  List outVals,
  List<int> begin,
  List<int> size,
) {
  int outOffset = 0;
  final beginI = begin[0];
  final beginJ = begin[1];
  final beginK = begin[2];
  final endI = beginI + size[0];
  final endJ = beginJ + size[1];
  for (int i = beginI; i < endI; i++) {
    for (int j = beginJ; j < endJ; j++) {
      final xOffset = i * xStride1 + j * xStride2 + beginK;
      outVals.setAll(outOffset, xVals.slice(xOffset, xOffset + size[2]));
      outOffset += size[2];
    }
  }
}

void slice4d(
  List xVals,
  int xStride1,
  int xStride2,
  int xStride3,
  List outVals,
  //: [number, number, number, number]
  List<int> begin,
  //: [number, number, number, number]
  List<int> size,
) {
  int outOffset = 0;
  final beginI = begin[0];
  final beginJ = begin[1];
  final beginK = begin[2];
  final endI = beginI + size[0];
  final endJ = beginJ + size[1];
  final endK = beginK + size[2];
  final beginL = begin[3];

  for (int i = beginI; i < endI; i++) {
    for (int j = beginJ; j < endJ; j++) {
      for (int k = beginK; k < endK; k++) {
        final xOffset = i * xStride1 + j * xStride2 + k * xStride3 + beginL;
        outVals.setAll(outOffset, xVals.slice(xOffset, xOffset + size[3]));
        outOffset += size[3];
      }
    }
  }
}

final sliceConfig = KernelConfigG(
  kernelName: Slice,
  backendName: 'wasm',
  kernelFunc: slice,
);
