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

// import {Keypoint} from './interfaces/common_interfaces';
// import {Rect} from './interfaces/shape_interfaces';

import 'dart:math' as Math;
import 'interfaces/common_interfaces.dart';
import 'interfaces/shape_interfaces.dart';

/**
 * Projects normalized landmarks in a rectangle to its original coordinates. The
 * rectangle must also be in normalized coordinates.
 * @param landmarks A normalized Landmark list representing landmarks in a
 *     normalized rectangle.
 * @param inputRect A normalized rectangle.
 * @param config Config object has one field ignoreRotation, default to false.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmark_projection_calculator.cc
List<Keypoint> calculateLandmarkProjection(
  List<Keypoint> landmarks,
  Rect inputRect, {
  bool ignoreRotation = false,
}) {
  final List<Keypoint> outputLandmarks = [];
  for (final landmark in landmarks) {
    final x = landmark.x - 0.5;
    final y = landmark.y - 0.5;
    final angle = ignoreRotation || inputRect.rotation == null
        ? 0.0
        : inputRect.rotation!;
    var newX = Math.cos(angle) * x - Math.sin(angle) * y;
    var newY = Math.sin(angle) * x + Math.cos(angle) * y;

    newX = newX * inputRect.width + inputRect.xCenter;
    newY = newY * inputRect.height + inputRect.yCenter;

    final newZ = landmark.z == null
        ? null
        : landmark.z! * inputRect.width; // Scale Z coordinate as x.

    final newLandmark = landmark.copyWith(
      x: newX,
      y: newY,
      z: Nullable(newZ),
    );

    outputLandmarks.add(newLandmark);
  }

  return outputLandmarks;
}
