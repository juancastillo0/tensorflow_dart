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

import 'dart:math' as math;

import '../tensor.dart' show Shape;

extension IndexOrNull<T> on List<T> {
  T? getIndexOrNull(int index) =>
      index >= length || index < 0 ? null : this[index];
}

/**
 * Returns the dimensions in the input shape that are broadcasted to
 * produce the provided output shape.
 *
 * The returned dimensions are 0-indexed and sorted. An example:
 * inShape = [4, 1, 3]
 * outShape = [5, 4, 3, 3]
 * result = [1]. Dimension 1 (2nd dimension of input) gets broadcasted 1 => 3.
 */
Shape getBroadcastDims(Shape inShape, Shape outShape) {
  final inRank = inShape.length;
  final Shape dims = [];
  for (int i = 0; i < inRank; i++) {
    final dim = inRank - 1 - i;
    final a = inShape.getIndexOrNull(dim) ?? 1;
    final b = outShape.getIndexOrNull(outShape.length - 1 - i) ?? 1;
    if (b > 1 && a == 1) {
      dims.insert(0, dim);
    }
  }
  return dims;
}

/**
 * Returns the axes in the output space that should be reduced to produce
 * the input space.
 */
Shape getReductionAxes(
  Shape inShape,
  Shape outShape,
) {
  final Shape result = [];
  for (int i = 0; i < outShape.length; i++) {
    final inDim = inShape.getIndexOrNull(inShape.length - i - 1);
    final outAxis = outShape.length - i - 1;
    final outDim = outShape[outAxis];
    if (inDim == null || (inDim == 1 && outDim > 1)) {
      result.insert(0, outAxis);
    }
  }
  return result;
}

Shape assertAndGetBroadcastShape(
  Shape shapeA,
  Shape shapeB,
) {
  final Shape result = [];
  final l = math.max(shapeA.length, shapeB.length);

  for (int i = 0; i < l; i++) {
    int? a = shapeA.getIndexOrNull(shapeA.length - i - 1);
    a ??= 1;

    int? b = shapeB.getIndexOrNull(shapeB.length - i - 1);
    b ??= 1;

    if (a == 1) {
      result.insert(0, b);
    } else if (b == 1) {
      result.insert(0, a);
    } else if (a != b) {
      final errMsg = 'Operands could not be broadcast together with shapes ' +
          '${shapeA} and ${shapeB}.';
      throw Exception(errMsg);
    } else {
      result.insert(0, a);
    }
  }
  return result;
}
