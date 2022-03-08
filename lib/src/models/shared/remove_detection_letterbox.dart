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

// import {Padding} from './interfaces/common_interfaces';
// import {Detection} from './interfaces/shape_interfaces';

import 'interfaces/common_interfaces.dart';
import 'interfaces/shape_interfaces.dart';

/**
 * Adjusts detection locations on the letterboxed image to the corresponding
 * locations on the same image with the letterbox removed (the input image to
 * the graph before image transformation).
 *
 * @param detections A list of detection boxes on an letterboxed image.
 * @param letterboxPadding A `padding` object representing the letterbox padding
 *     from the 4 sides: left, top, right, bottom, of the letterboxed image,
 *     normalized by the letterboxed image dimensions.
 * @returns detections: A list of detection boxes representing detections with
 *     their locations adjusted to the letterbox-removed (non-padded) image.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detection_letterbox_removal_calculator.cc
List<Detection> removeDetectionLetterbox(
    List<Detection> detections, Padding letterboxPadding) {
  final left = letterboxPadding.left;
  final top = letterboxPadding.top;
  final leftAndRight = letterboxPadding.left + letterboxPadding.right;
  final topAndBottom = letterboxPadding.top + letterboxPadding.bottom;

  for (int i = 0; i < detections.length; i++) {
    final detection = detections[i];
    final relativeBoundingBox = detection.locationData.relativeBoundingBox;
    final xMin = (relativeBoundingBox.xMin - left) / (1 - leftAndRight);
    final yMin = (relativeBoundingBox.yMin - top) / (1 - topAndBottom);
    final width = relativeBoundingBox.width / (1 - leftAndRight);
    final height = relativeBoundingBox.height / (1 - topAndBottom);
    relativeBoundingBox.xMin = xMin;
    relativeBoundingBox.yMin = yMin;
    relativeBoundingBox.width = width;
    relativeBoundingBox.height = height;
    relativeBoundingBox.xMax = xMin + width;
    relativeBoundingBox.yMax = yMin + height;

    final relativeKeypoints = detection.locationData.relativeKeypoints;

    if (relativeKeypoints) {
      relativeKeypoints.forEach((keypoint) {
        final newX = (keypoint.x - left) / (1 - leftAndRight);
        final newY = (keypoint.y - top) / (1 - topAndBottom);
        keypoint.x = newX;
        keypoint.y = newY;
      });
    }
  }

  return detections;
}
