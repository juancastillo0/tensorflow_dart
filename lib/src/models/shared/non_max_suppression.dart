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
// import * as tf from '@tensorflow/tfjs-core';
// import {Detection} from './interfaces/shape_interfaces';

import 'dart:math' as Math;
import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'interfaces/shape_interfaces.dart';

Future<List<Detection>> nonMaxSuppression(
  List<Detection> detections,
  int maxDetections,
  double iouThreshold,
  // Currently only IOU overap is supported.
  // overlapType: 'intersection-over-union',
) async {
  // Sort to match NonMaxSuppresion calculator's decreasing detection score
  // traversal.
  // NonMaxSuppresionCalculator: RetainMaxScoringLabelOnly
  detections.sort((detectionA, detectionB) =>
      (detectionB.score!.reduce(Math.max) - detectionA.score!.reduce(Math.max))
          .round());

  final detectionsTensor = tf.tensor2d(detections.map((d) => [
        d.locationData.relativeBoundingBox.yMin,
        d.locationData.relativeBoundingBox.xMin,
        d.locationData.relativeBoundingBox.yMax,
        d.locationData.relativeBoundingBox.xMax
      ]));
  final scoresTensor = tf.tensor1d(detections.map((d) => d.score![0]));

  final selectedIdsTensor = await tf.image.nonMaxSuppressionAsync(
      detectionsTensor, scoresTensor, maxDetections,
      iouThreshold: iouThreshold);
  final selectedIds = await selectedIdsTensor.array() as List;

  final selectedDetections =
      detections.whereIndexed((i, _) => selectedIds.contains(i)).toList();

  tf.dispose([detectionsTensor, scoresTensor, selectedIdsTensor]);

  return selectedDetections;
}
