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

// import {backend_util, KernelConfig, KernelFunc, Mean, MeanAttrs, MeanInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {cast} from './Cast';

// import {permuteAxesAndTranspose} from './kernel_utils';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'cast.dart';
import 'kernel_utils.dart';

late final Function(List) _wasmMean;
// : (xId: number, reduceSize: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmMean =
      backend.wasm.cwrap(Mean, null /*void*/, ['number, number, number']);
}

TensorInfo mean({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final axis =
      (attrs!['axis'] is int ? [attrs['axis']] : attrs['axis']) as List<int>;
  final keepDims = attrs['keepDims'] as bool;
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  var inputId = xId;
  var input = x;

  final _p = permuteAxesAndTranspose(x, axis, backend);
  final transposed = _p.transposed;
  final axes = _p.axes;
  final originalAxes = _p.originalAxes;
  final inputWasTransposed = _p.inputWasTransposed;

  var reductionAxes = axes;
  if (inputWasTransposed) {
    final transposedId = backend.dataIdMap.get(transposed!.dataId)!.id;
    if (transposedId != xId) {
      // transpose was not a no-op. We will need to dispose of this
      // once we are done.
      input = transposed;
      inputId = transposedId;
      reductionAxes = backend_util.getInnerMostAxes(
          reductionAxes.length, input.shape.length);
    }
  }

  backend_util.assertAxesAreInnerMostDims(
      'mean', reductionAxes, input.shape.length);
  final _shapes =
      backend_util.computeOutAndReduceShapes(input.shape, reductionAxes);
  final outShape = _shapes.outShape;
  final reduceShape = _shapes.reduceShape;
  final reduceSize = util.sizeFromShape(reduceShape);
  var castedInput = input;
  if (input.dtype != 'float32') {
    castedInput = cast(
        backend: backend, inputs: {'x': input}, attrs: {'dtype': 'float32'});
    inputId = backend.dataIdMap.get(castedInput.dataId)!.id;
  }

  final out = backend.makeOutput(outShape, 'float32');
  if (util.sizeFromShape(input.shape) != 0) {
    final outId = backend.dataIdMap.get(out.dataId)!.id;
    _wasmMean([inputId, reduceSize, outId]);
  }

  if (inputWasTransposed) {
    // dispose of the transposed tensor.
    backend.disposeData(transposed!.dataId);
  }

  if (input.dtype != 'float32') {
    backend.disposeData(castedInput.dataId);
  }

  if (keepDims) {
    // reshape
    final newShape = backend_util.expandShapeToKeepDim(out.shape, originalAxes);
    return copyTensorInfo(out, shape: newShape);
  }

  return out;
}

final meanConfig = KernelConfigG(
  kernelName: Mean,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: mean,
);
