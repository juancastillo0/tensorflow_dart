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
 * Projects world landmarks from the rectangle to original coordinates.
 *
 * World landmarks are predicted in meters rather than in pixels of the image
 * and have origin in the middle of the hips rather than in the corner of the
 * pose image (cropped with given rectangle). Thus only rotation (but not scale
 * and translation) is applied to the landmarks to transform them back to
 * original coordinates.
 * @param worldLandmarks A Landmark list representing world landmarks in the
 *     rectangle.
 * @param inputRect A normalized rectangle.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmark_projection_calculator.cc
List<Keypoint> calculateWorldLandmarkProjection(
  List<Keypoint> worldLandmarks,
  Rect inputRect,
) {
  final outputLandmarks = <Keypoint>[];
  for (final worldLandmark in worldLandmarks) {
    final x = worldLandmark.x;
    final y = worldLandmark.y;
    final angle = inputRect.rotation;
    final newX = Math.cos(angle) * x - Math.sin(angle) * y;
    final newY = Math.sin(angle) * x + Math.cos(angle) * y;

    outputLandmarks.add(worldLandmark.copyWith(
      x: newX,
      y: newY,
    ));
  }

  return outputLandmarks;
}
