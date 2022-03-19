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

// import {Matrix4x4, matrix4x4ToArray} from './calculate_inverse_matrix';
// import {Detection} from './interfaces/shape_interfaces';

import 'dart:math' as Math;
import 'calculate_inverse_matrix.dart';
import 'interfaces/shape_interfaces.dart';

List<double> _project(List<double> projectionMatrix,
    {required double x, required double y}) {
  return [
    x * projectionMatrix[0] + y * projectionMatrix[1] + projectionMatrix[3],
    x * projectionMatrix[4] + y * projectionMatrix[5] + projectionMatrix[7]
  ];
}

/**
 * Projects detections to a different coordinate system using a provided
 * projection matrix.
 *
 * @param detections A list of detections to project using the provided
 *     projection matrix.
 * @param projectionMatrix Maps data from one coordinate system to     another.
 * @returns detections: A list of projected detections
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detection_projection_calculator.cc
List<Detection> detectionProjection(
  List<Detection> detections,
  Matrix4x4 projectionMatrix,
) {
  final flatProjectionMatrix = matrix4x4ToArray(projectionMatrix);

  return detections.map((detection) {
    final locationData = detection.locationData;

    // Project keypoints.
    final relativeKeypoints = locationData.relativeKeypoints.map((keypoint) {
      final _l = _project(flatProjectionMatrix, x: keypoint.x, y: keypoint.y);
      return keypoint.copyWith(
        x: _l[0],
        y: _l[1],
      );
    }).toList();

    // Project bounding box.
    final box = locationData.relativeBoundingBox;

    var xMin = double.maxFinite,
        yMin = double.maxFinite,
        xMax = double.minPositive,
        yMax = double.minPositive;

    [
      [box.xMin, box.yMin],
      [box.xMin + box.width, box.yMin],
      [box.xMin + box.width, box.yMin + box.height],
      [box.xMin, box.yMin + box.height]
    ].forEach((coordinate) {
      // a) Define and project box points.
      final _l =
          _project(flatProjectionMatrix, x: coordinate[0], y: coordinate[1]);
      final x = _l[0];
      final y = _l[1];
      // b) Find new left top and right bottom points for a box which
      // encompases
      //    non-projected (rotated) box.
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    });
    return detection.copyWith(
      locationData: LocationData(
        relativeKeypoints: relativeKeypoints,
        relativeBoundingBox: BoundingBox(
          xMin: xMin,
          xMax: xMax,
          yMin: yMin,
          yMax: yMax,
          width: xMax - xMin,
          height: yMax - yMin,
        ),
      ),
    );
  }).toList();
}
