/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

// import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';
// import {BackendWasm} from '../backend_wasm';
// import {transpose} from './Transpose';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

import 'transpose.dart';

class PermutedTensorInfo {
  final TensorInfo? transposed;
  final List<int> axes;
  final List<int> originalAxes;
  final bool inputWasTransposed;

  PermutedTensorInfo({
    this.transposed,
    required this.axes,
    required this.originalAxes,
    required this.inputWasTransposed,
  });
}

/**
 * Compute permutation axes and do a transpose if necessary.
 *
 * Used by reduction ops.
 * @param x input TensorInfo
 * @param axis reduction axes
 * @param backend wasm backend instance
 */
PermutedTensorInfo permuteAxesAndTranspose(
  TensorInfo x,
  // : number|number[]
  List<int> axis,
  BackendWasm backend,
) {
  final xShape = x.shape;
  final xRank = x.shape.length;

  final originalAxes = util.parseAxisParam(axis, xShape);
  var axes = originalAxes;
  final permutedAxes = backend_util.getAxesPermutation(axes, xRank);
  var xTransposed = null;
  var inputWasTransposed = false;
  if (permutedAxes != null) {
    final List<int> newShape =
        List.generate(xRank, (i) => xShape[permutedAxes[i]]);

    axes = backend_util.getInnerMostAxes(axes.length, xRank);
    xTransposed = transpose(
      inputs: {'x': x},
      attrs: {'perm': permutedAxes},
      backend: backend,
    );

    final xId = backend.dataIdMap.get(x.dataId)!.id;
    final transposedId = backend.dataIdMap.get(xTransposed.dataId)!.id;
    if (transposedId != xId) {
      inputWasTransposed = true;
    }
  }

  return PermutedTensorInfo(
    transposed: xTransposed,
    originalAxes: originalAxes,
    axes: axes,
    inputWasTransposed: inputWasTransposed,
  );
}
