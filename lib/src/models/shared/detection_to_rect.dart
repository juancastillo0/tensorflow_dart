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

// import {normalizeRadians} from './image_utils';
// import {ImageSize} from './interfaces/common_interfaces';
// import {DetectionToRectConfig} from './interfaces/config_interfaces';
// import {BoundingBox, Detection, LocationData, Rect} from './interfaces/shape_interfaces';
import 'dart:math' as Math;

import 'image_utils.dart';
import 'interfaces/common_interfaces.dart';
import 'interfaces/config_interfaces.dart';
import 'interfaces/shape_interfaces.dart';

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detections_to_rects_calculator.cc
double computeRotation(
  Detection detection,
  ImageSize imageSize,
  DetectionToRectConfig config,
) {
  final locationData = detection.locationData;
  final startKeypoint = config.rotationVectorStartKeypointIndex;
  final endKeypoint = config.rotationVectorEndKeypointIndex;

  final double targetAngle = config.rotationVectorTargetAngle ??
      Math.pi * config.rotationVectorTargetAngleDegree! / 180;

  final relativeKeypoints = locationData.relativeKeypoints;
  final x0 = relativeKeypoints[startKeypoint].x * imageSize.width;
  final y0 = relativeKeypoints[startKeypoint].y * imageSize.height;
  final x1 = relativeKeypoints[endKeypoint].x * imageSize.width;
  final y1 = relativeKeypoints[endKeypoint].y * imageSize.height;

  final rotation =
      normalizeRadians(targetAngle - Math.atan2(-(y1 - y0), x1 - x0));

  return rotation;
}

Rect _rectFromBox(BoundingBox box) {
  return Rect(
    xCenter: box.xMin + box.width / 2,
    yCenter: box.yMin + box.height / 2,
    width: box.width,
    height: box.height,
  );
}

Rect _normRectFromKeypoints(LocationData locationData) {
  final keypoints = locationData.relativeKeypoints;
  if (keypoints.length <= 1) {
    throw Exception('2 or more keypoints required to calculate a rect.');
  }
  var xMin = double.maxFinite,
      yMin = double.maxFinite,
      xMax = double.minPositive,
      yMax = double.minPositive;

  keypoints.forEach((keypoint) {
    xMin = Math.min(xMin, keypoint.x);
    xMax = Math.max(xMax, keypoint.x);
    yMin = Math.min(yMin, keypoint.y);
    yMax = Math.max(yMax, keypoint.y);
  });

  return Rect(
    xCenter: (xMin + xMax) / 2,
    yCenter: (yMin + yMax) / 2,
    width: xMax - xMin,
    height: yMax - yMin,
  );
}

Rect _detectionToNormalizedRect(
    Detection detection, ConversionMode conversionMode) {
  final locationData = detection.locationData;
  return conversionMode == ConversionMode.boundingbox
      ? _rectFromBox(locationData.relativeBoundingBox)
      : _normRectFromKeypoints(locationData);
}

Rect _detectionToRect(
  Detection detection,
  ConversionMode conversionMode,
  ImageSize? imageSize,
) {
  final locationData = detection.locationData;

  if (conversionMode == ConversionMode.boundingbox) {
    return _rectFromBox(locationData.boundingBox!);
  } else {
    final rect = _normRectFromKeypoints(locationData);
    final width = imageSize!.width;
    final height = imageSize.height;

    return Rect(
      xCenter: rect.xCenter * width,
      yCenter: rect.yCenter * height,
      width: rect.width * width,
      height: rect.height * height,
    );
  }
}

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detections_to_rects_calculator.cc
Rect calculateDetectionsToRects(
  Detection detection,
  ConversionMode conversionMode, {
  required bool normalized,
  required ImageSize imageSize,
  DetectionToRectConfig? rotationConfig,
}) {
  final rect = !normalized
      ? _detectionToRect(detection, conversionMode, imageSize)
      : _detectionToNormalizedRect(detection, conversionMode);

  if (rotationConfig != null) {
    return rect.copyWith(
      rotation: computeRotation(detection, imageSize, rotationConfig),
    );
  }

  return rect;
}

enum ConversionMode {
  boundingbox,
  keypoints,
}
