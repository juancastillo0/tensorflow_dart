/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import 'dart:math' as Math;

typedef TransformationMatrix = List<List<double>>;
// [
//   [number, number, number], [number, number, number], [number, number, number]
// ];

double normalizeRadians(double angle) {
  return angle - 2 * Math.pi * ((angle + Math.pi) / (2 * Math.pi)).floor();
}

double computeRotation(List<double> point1, List<double> point2) {
  final radians =
      Math.pi / 2 - Math.atan2(-(point2[1] - point1[1]), point2[0] - point1[0]);
  return normalizeRadians(radians);
}

TransformationMatrix _buildTranslationMatrix(double x, double y) => ([
      [1, 0, x],
      [0, 1, y],
      [0, 0, 1]
    ]);

double dot(List<double> v1, List<double> v2) {
  double product = 0;
  for (int i = 0; i < v1.length; i++) {
    product += v1[i] * v2[i];
  }
  return product;
}

List<double> getColumnFrom2DArr(List<List<double>> arr, int columnIndex) {
  final List<double> column = [];

  for (int i = 0; i < arr.length; i++) {
    column.add(arr[i][columnIndex]);
  }

  return column;
}

TransformationMatrix _multiplyTransformMatrices(
    List<List<double>> mat1, List<List<double>> mat2) {
  final TransformationMatrix product = [];

  final size = mat1.length;

  for (int row = 0; row < size; row++) {
    product.add([]);
    for (int col = 0; col < size; col++) {
      product[row].add(dot(mat1[row], getColumnFrom2DArr(mat2, col)));
    }
  }

  return product;
}

TransformationMatrix buildRotationMatrix(double rotation, List<double> center) {
  final cosA = Math.cos(rotation);
  final sinA = Math.sin(rotation);

  final List<List<double>> rotationMatrix = [
    [cosA, -sinA, 0],
    [sinA, cosA, 0],
    [0, 0, 1]
  ];
  final translationMatrix = _buildTranslationMatrix(center[0], center[1]);
  final translationTimesRotation =
      _multiplyTransformMatrices(translationMatrix, rotationMatrix);

  final negativeTranslationMatrix =
      _buildTranslationMatrix(-center[0], -center[1]);
  return _multiplyTransformMatrices(
      translationTimesRotation, negativeTranslationMatrix);
}

TransformationMatrix invertTransformMatrix(TransformationMatrix matrix) {
  final rotationComponent = [
    [matrix[0][0], matrix[1][0]],
    [matrix[0][1], matrix[1][1]]
  ];
  final translationComponent = [matrix[0][2], matrix[1][2]];
  final invertedTranslation = [
    -dot(rotationComponent[0], translationComponent),
    -dot(rotationComponent[1], translationComponent)
  ];

  return [
    [...rotationComponent[0], invertedTranslation[0]],
    [...rotationComponent[1], invertedTranslation[1]],
    [0, 0, 1]
  ];
}

List<double> rotatePoint(
  List<double> homogeneousCoordinate,
  TransformationMatrix rotationMatrix,
) {
  return [
    dot(homogeneousCoordinate, rotationMatrix[0]),
    dot(homogeneousCoordinate, rotationMatrix[1])
  ];
}
