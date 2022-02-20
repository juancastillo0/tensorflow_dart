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

// import {ENGINE} from '../engine';
// import {Tile, TileAttrs, TileInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {Rank, ShapeMap, TensorLike} from '../types';

// import {clone} from './clone';
// import {op} from './operation';
// import {reshape} from './reshape';

import '_prelude.dart';
import 'clone.dart' show clone;
import 'reshape.dart' show reshape;
import 'package:collection/collection.dart';

/**
 * Broadcast an array to a compatible shape NumPy-style.
 *
 * The tensor's shape is compared to the broadcast shape from end to beginning.
 * Ones are prepended to the tensor's shape until is has the same length as
 * the broadcast shape. If input.shape[i]==shape[i], the (i+1)-th axis is
 * already broadcast-compatible. If input.shape[i]==1 and shape[i]==N, then
 * the input tensor is tiled N times along that axis (using tf.tile).
 *
 * @param input The tensor that is to be broadcasted.
 * @param shape The input is to be broadcast to this shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
Tensor<R> broadcastTo<R extends Rank>(Tensor x, Shape shape) {
  return execOp('broadcastTo', () {
    var input = convertToTensor(x, 'broadcastTo', 'x');
    final xShape = input.shape;

    if (shape.any((d) => !(d > 0) || d % 1 != 0)) {
      throw Exception('broadcastTo(): Invalid broadcast shape [${shape}].');
    }

    if (shape.length < input.rank) {
      throw Exception(
          'broadcastTo(): shape.length=${shape.length} < input.rank=${input.rank}.');
    }

    if (shape.length > input.rank) {
      final newShape = [...input.shape];
      while (newShape.length < shape.length) {
        newShape.insert(0, 1);
      }
      input = reshape(input, newShape);
    }

    final inputShape = input.shape;
    final List<int> reps = [...shape];
    for (int i = shape.length - 1; i >= 0; i--) {
      if (inputShape[i] == shape[i]) {
        reps[i] = 1;
      } else if (input.shape[i] != 1) {
        throw Exception(
            'broadcastTo(): [${xShape}] cannot be broadcast to [${shape}].');
      }
    }
    final axes = reps.mapIndexed((i, n) => n > 1 ? i : -1).where((i) => i >= 0);

    if (axes.isEmpty) {
      return clone(input) as Tensor<R>;
    }

    // TODO call broadcastTo kernel directly once backends implement broadcstTo
    final inputs = {'x': input}; // : TileInputs
    final attrs = {'reps': reps}; // : TileAttrs
    return ENGINE.runKernel(Tile, inputs, attrs) as Tensor<R>;
  });
}
