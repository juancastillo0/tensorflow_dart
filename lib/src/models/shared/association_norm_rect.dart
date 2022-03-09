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

// import {BoundingBox, Rect} from './interfaces/shape_interfaces';
import 'dart:math' as Math;
import 'interfaces/shape_interfaces.dart';

double _area(BoundingBox rect) {
  return rect.width * rect.height;
}

bool _intersects(BoundingBox rect1, BoundingBox rect2) {
  return !(rect1.xMax < rect2.xMin ||
      rect2.xMax < rect1.xMin ||
      rect1.yMax < rect2.yMin ||
      rect2.yMax < rect1.yMin);
}

BoundingBox _intersect(BoundingBox rect1, BoundingBox rect2) {
  final xMin = Math.max(rect1.xMin, rect2.xMin);
  final xMax = Math.min(rect1.xMax, rect2.xMax);
  final yMin = Math.max(rect1.yMin, rect2.yMin);
  final yMax = Math.min(rect1.yMax, rect2.yMax);
  final width = Math.max(xMax - xMin, 0.0);
  final height = Math.max(yMax - yMin, 0.0);

  return BoundingBox(
    xMin: xMin,
    xMax: xMax,
    yMin: yMin,
    yMax: yMax,
    width: width,
    height: height,
  );
}

BoundingBox _getBoundingBox(Rect rect) {
  final xMin = rect.xCenter - rect.width / 2;
  final xMax = xMin + rect.width;
  final yMin = rect.yCenter - rect.height / 2;
  final yMax = yMin + rect.height;

  return BoundingBox(
    xMin: xMin,
    xMax: xMax,
    yMin: yMin,
    yMax: yMax,
    width: rect.width,
    height: rect.height,
  );
}

double overlapSimilarity(Rect rect1, Rect rect2) {
  final bbox1 = _getBoundingBox(rect1);
  final bbox2 = _getBoundingBox(rect2);
  if (!_intersects(bbox1, bbox2)) {
    return 0;
  }
  final intersectionArea = _area(_intersect(bbox1, bbox2));
  final normalization = _area(bbox1) + _area(bbox2) - intersectionArea;
  return normalization > 0 ? intersectionArea / normalization : 0;
}

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/association_norm_rect_calculator.cc
// Propgating ids from previous to current is not performed by this code.
List<Rect> calculateAssociationNormRect(
  List<List<Rect>> rectsArray,
  double minSimilarityThreshold,
) {
  List<Rect> result = [];

  // rectsArray elements are interpreted to be sorted in reverse priority order,
  // so later elements are higher in priority. This means that if there's a
  // large overlap, the later rect will be added and the older rect will be
  // removed.
  rectsArray.forEach((rects) => rects.forEach((curRect) {
        result = result
            .where((prevRect) =>
                overlapSimilarity(curRect, prevRect) <= minSimilarityThreshold)
            .toList();
        result.add(curRect);
      }));

  return result;
}
