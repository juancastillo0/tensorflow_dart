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

// import {normalizeRadians} from '../../shared/calculators/image_utils';
// import {ImageSize, Keypoint} from '../../shared/calculators/interfaces/common_interfaces';
// import  {Rect} from '../../shared/calculators/interfaces/shape_interfaces';

import '../../../shared/interfaces/common_interfaces.dart';
import '../../../shared/interfaces/shape_interfaces.dart';

import 'dart:math' as Math;

const WRIST_JOINT = 0;
const MIDDLE_FINGER_PIP_JOINT = 6;
const INDEX_FINGER_PIP_JOINT = 4;
const RING_FINGER_PIP_JOINT = 8;

double _computeRotation(List<Keypoint> landmarks, ImageSize imageSize) {
  final x0 = landmarks[WRIST_JOINT].x * imageSize.width;
  final y0 = landmarks[WRIST_JOINT].y * imageSize.height;

  var x1 = (landmarks[INDEX_FINGER_PIP_JOINT].x +
          landmarks[RING_FINGER_PIP_JOINT].x) /
      2;
  var y1 = (landmarks[INDEX_FINGER_PIP_JOINT].y +
          landmarks[RING_FINGER_PIP_JOINT].y) /
      2;
  x1 = (x1 + landmarks[MIDDLE_FINGER_PIP_JOINT].x) / 2 * imageSize.width;
  y1 = (y1 + landmarks[MIDDLE_FINGER_PIP_JOINT].y) / 2 * imageSize.height;

  final rotation =
      normalizeRadians(Math.pi / 2 - Math.atan2(-(y1 - y0), x1 - x0));
  return rotation;
}

/**
 * @param landmarks List of normalized landmarks.
 *
 * @returns A `Rect`.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc;bpv=1
Rect handLandmarksToRect(List<Keypoint> landmarks, ImageSize imageSize) {
  final rotation = _computeRotation(landmarks, imageSize);
  final reverseAngle = normalizeRadians(-rotation);

  var minX = double.infinity,
      maxX = double.negativeInfinity,
      minY = double.infinity,
      maxY = double.negativeInfinity;

  // Find boundaries of landmarks.
  for (final landmark in landmarks) {
    final x = landmark.x;
    final y = landmark.y;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  final axisAlignedcenterX = (maxX + minX) / 2;
  final axisAlignedcenterY = (maxY + minY) / 2;

  minX = double.infinity;
  maxX = double.negativeInfinity;
  minY = double.infinity;
  maxY = double.negativeInfinity;
  // Find boundaries of rotated landmarks.
  for (final landmark in landmarks) {
    final originalX = (landmark.x - axisAlignedcenterX) * imageSize.width;
    final originalY = (landmark.y - axisAlignedcenterY) * imageSize.height;

    final projectedX =
        originalX * Math.cos(reverseAngle) - originalY * Math.sin(reverseAngle);
    final projectedY =
        originalX * Math.sin(reverseAngle) + originalY * Math.cos(reverseAngle);

    minX = Math.min(minX, projectedX);
    maxX = Math.max(maxX, projectedX);
    minY = Math.min(minY, projectedY);
    maxY = Math.max(maxY, projectedY);
  }

  final projectedCenterX = (maxX + minX) / 2;
  final projectedCenterY = (maxY + minY) / 2;

  final centerX = projectedCenterX * Math.cos(rotation) -
      projectedCenterY * Math.sin(rotation) +
      imageSize.width * axisAlignedcenterX;
  final centerY = projectedCenterX * Math.sin(rotation) +
      projectedCenterY * Math.cos(rotation) +
      imageSize.height * axisAlignedcenterY;
  final width = (maxX - minX) / imageSize.width;
  final height = (maxY - minY) / imageSize.height;

  return Rect(
    xCenter: centerX / imageSize.width,
    yCenter: centerY / imageSize.height,
    width: width,
    height: height,
    rotation: rotation,
  );
}
