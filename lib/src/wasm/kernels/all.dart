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

// import {All, AllAttrs, AllInputs, backend_util, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {permuteAxesAndTranspose} from './kernel_utils';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'kernel_utils.dart';

late final Function(List)
    _wasmAll; //: (xId: number, reduceSize: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmAll = backend.wasm.cwrap(All, null /*void*/, ['number, number, number']);
}

TensorInfo all({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final axis =
      (attrs!['axis'] is int ? [attrs['axis']] : attrs['axis']) as List<int>;
  final keepDims = attrs['keepDims'] as bool;

  final x = inputs['x']!;
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  var inputId = xId;
  var input = x;

  final _p = permuteAxesAndTranspose(x, axis, backend);
  final transposed = _p.transposed;
  final axes = _p.axes;
  final originalAxes = _p.originalAxes;
  final inputWasTransposed = _p.inputWasTransposed;

  if (inputWasTransposed) {
    final transposedId = backend.dataIdMap.get(transposed!.dataId)!.id;
    input = transposed;
    inputId = transposedId;
  }

  final inputRank = input.shape.length;
  backend_util.assertAxesAreInnerMostDims('all', axes, inputRank);
  final _shapes = backend_util.computeOutAndReduceShapes(input.shape, axes);
  final outShape = _shapes.outShape;
  final reduceShape = _shapes.reduceShape;
  final reduceSize = util.sizeFromShape(reduceShape);

  final out = backend.makeOutput(outShape, x.dtype);
  if (util.sizeFromShape(input.shape) != 0) {
    final outId = backend.dataIdMap.get(out.dataId)!.id;
    _wasmAll([inputId, reduceSize, outId]);
  }

  if (inputWasTransposed) {
    // dispose of the transposed tensor.
    backend.disposeData(transposed!.dataId);
  }

  if (keepDims) {
    // reshape
    final newShape = backend_util.expandShapeToKeepDim(out.shape, originalAxes);
    return copyTensorInfo(out, shape: newShape);
  }

  return out;
}

final allConfig = KernelConfigG(
    kernelName: All, backendName: 'wasm', setupFunc: _setup, kernelFunc: all);
