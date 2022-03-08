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
// import {RectTransformationConfig} from './interfaces/config_interfaces';
// import {Rect} from './interfaces/shape_interfaces';

import 'dart:math' as Math;

import 'image_utils.dart';
import 'interfaces/common_interfaces.dart';
import 'interfaces/config_interfaces.dart';
import 'interfaces/shape_interfaces.dart';

/**
 * Performs geometric transformation to the input normalized rectangle,
 * correpsonding to input normalized rectangle respectively.
 * @param rect The normalized rectangle.
 * @param imageSize The original imageSize.
 * @param config See documentation in `RectTransformationConfig`.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc
Rect transformNormalizedRect(
  Rect rect,
  ImageSize imageSize,
  RectTransformationConfig config,
) {
  var width = rect.width;
  var height = rect.height;
  var rotation = rect.rotation;

  if (config.rotation != null || config.rotationDegree != null) {
    rotation = computeNewRotation(rotation, config);
  }

  if (rotation == 0) {
    rect.xCenter = rect.xCenter + width * config.shiftX;
    rect.yCenter = rect.yCenter + height * config.shiftY;
  } else {
    final xShift = (imageSize.width *
                width *
                config.shiftX *
                Math.cos(rotation) -
            imageSize.height * height * config.shiftY * Math.sin(rotation)) /
        imageSize.width;
    final yShift = (imageSize.width *
                width *
                config.shiftX *
                Math.sin(rotation) +
            imageSize.height * height * config.shiftY * Math.cos(rotation)) /
        imageSize.height;
    rect.xCenter = rect.xCenter + xShift;
    rect.yCenter = rect.yCenter + yShift;
  }

  if (config.squareLong == true) {
    final longSide =
        Math.max(width * imageSize.width, height * imageSize.height);
    width = longSide / imageSize.width;
    height = longSide / imageSize.height;
  } else if (config.squareShort == true) {
    final shortSide =
        Math.min(width * imageSize.width, height * imageSize.height);
    width = shortSide / imageSize.width;
    height = shortSide / imageSize.height;
  }
  rect.width = width * config.scaleX;
  rect.height = height * config.scaleY;

  return rect;
}

double computeNewRotation(double rotation, RectTransformationConfig config) {
  if (config.rotation != null) {
    rotation += config.rotation!;
  } else if (config.rotationDegree != null) {
    rotation += Math.pi * config.rotationDegree! / 180;
  }
  return normalizeRadians(rotation);
}
