/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

// import * as util from '../util';
import '../util_base.dart' as util;
import 'package:collection/collection.dart';

assertParamsConsistent(List<List<int>> shapes, int axis) {
  final rank = shapes[0].length;
  shapes.forEachIndexed((i, shape) {
    util.assert_(
        shape.length == rank,
        () =>
            'Error in concat${rank}D: rank of tensors[${i}] must be the same ' +
            'as the rank of the rest (${rank})');
  });

  util.assert_(axis >= 0 && axis < rank,
      () => 'Error in concat${rank}D: axis must be between 0 and ${rank - 1}.');

  final firstShape = shapes[0];
  shapes.forEachIndexed((i, shape) {
    for (int r = 0; r < rank; r++) {
      util.assert_(
          (r == axis) || (shape[r] == firstShape[r]),
          () =>
              'Error in concat${rank}D: Shape of tensors[${i}] (${shape}) ' +
              'does not match the shape of the rest (${firstShape}) ' +
              'along the non-concatenated axis ${i}.');
    }
  });
}

List<int> computeOutShape(List<List<int>> shapes, int axis) {
  final outputShape = [...shapes[0]];
  for (int i = 1; i < shapes.length; i++) {
    outputShape[axis] += shapes[i][axis];
  }
  return outputShape;
}
