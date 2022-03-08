/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

typedef Matrix4x4Row = List<double>; // [number, number, number, number];
typedef Matrix4x4 = List<Matrix4x4Row>;
// [Matrix4x4Row, Matrix4x4Row, Matrix4x4Row, Matrix4x4Row];

List<double> matrix4x4ToArray(Matrix4x4 matrix) {
  return matrix.expand((element) => element).toList();
}

Matrix4x4 arrayToMatrix4x4(List<double> array) {
  if (array.length != 16) {
    throw Exception('Array length must be 16 but got ${array.length}');
  }
  return [
    [array[0], array[1], array[2], array[3]],
    [array[4], array[5], array[6], array[7]],
    [array[8], array[9], array[10], array[11]],
    [array[12], array[13], array[14], array[15]],
  ];
}

double _generalDet3Helper(
    Matrix4x4 matrix, int i1, int i2, int i3, int j1, int j2, int j3) {
  return matrix[i1][j1] *
      (matrix[i2][j2] * matrix[i3][j3] - matrix[i2][j3] * matrix[i3][j2]);
}

double _cofactor4x4(Matrix4x4 matrix, int i, int j) {
  final i1 = (i + 1) % 4,
      i2 = (i + 2) % 4,
      i3 = (i + 3) % 4,
      j1 = (j + 1) % 4,
      j2 = (j + 2) % 4,
      j3 = (j + 3) % 4;
  return _generalDet3Helper(matrix, i1, i2, i3, j1, j2, j3) +
      _generalDet3Helper(matrix, i2, i3, i1, j1, j2, j3) +
      _generalDet3Helper(matrix, i3, i1, i2, j1, j2, j3);
}

/**
 * Calculates inverse of an invertible 4x4 matrix.
 * @param matrix 4x4 matrix to invert.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/inverse_matrix_calculator.cc
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/LU/InverseImpl.h
Matrix4x4 calculateInverseMatrix(Matrix4x4 matrix) {
  final inverse = arrayToMatrix4x4(List.filled(16, 0));

  inverse[0][0] = _cofactor4x4(matrix, 0, 0);
  inverse[1][0] = -_cofactor4x4(matrix, 0, 1);
  inverse[2][0] = _cofactor4x4(matrix, 0, 2);
  inverse[3][0] = -_cofactor4x4(matrix, 0, 3);
  inverse[0][2] = _cofactor4x4(matrix, 2, 0);
  inverse[1][2] = -_cofactor4x4(matrix, 2, 1);
  inverse[2][2] = _cofactor4x4(matrix, 2, 2);
  inverse[3][2] = -_cofactor4x4(matrix, 2, 3);
  inverse[0][1] = -_cofactor4x4(matrix, 1, 0);
  inverse[1][1] = _cofactor4x4(matrix, 1, 1);
  inverse[2][1] = -_cofactor4x4(matrix, 1, 2);
  inverse[3][1] = _cofactor4x4(matrix, 1, 3);
  inverse[0][3] = -_cofactor4x4(matrix, 3, 0);
  inverse[1][3] = _cofactor4x4(matrix, 3, 1);
  inverse[2][3] = -_cofactor4x4(matrix, 3, 2);
  inverse[3][3] = _cofactor4x4(matrix, 3, 3);

  final scale = matrix[0][0] * inverse[0][0] +
      matrix[1][0] * inverse[0][1] +
      matrix[2][0] * inverse[0][2] +
      matrix[3][0] * inverse[0][3];

  for (int i = 0; i < inverse.length; i++) {
    for (int j = 0; j < inverse.length; j++) {
      inverse[i][j] /= scale;
    }
  }
  return inverse;
}
