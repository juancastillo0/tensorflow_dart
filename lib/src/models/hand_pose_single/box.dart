/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import 'dart:math' as Math;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

// The hand bounding box.
class Box {
  final List<double> startPoint;
  final List<double> endPoint;
  final List<List<double>>? palmLandmarks;

  Box({
    required this.startPoint,
    required this.endPoint,
    this.palmLandmarks,
  });
}

List<double> getBoxSize(Box box) {
  return [
    (box.endPoint[0] - box.startPoint[0]).abs(),
    (box.endPoint[1] - box.startPoint[1]).abs(),
  ];
}

List<double> getBoxCenter(Box box) {
  return [
    box.startPoint[0] + (box.endPoint[0] - box.startPoint[0]) / 2,
    box.startPoint[1] + (box.endPoint[1] - box.startPoint[1]) / 2
  ];
}

tf.Tensor4D cutBoxFromImageAndResize(
    Box box, tf.Tensor4D image, List<int> cropSize) {
  final h = image.shape[1];
  final w = image.shape[2];

  final boxes = [
    [
      box.startPoint[1] / h,
      box.startPoint[0] / w,
      box.endPoint[1] / h,
      box.endPoint[0] / w
    ]
  ];

  return tf.image.cropAndResize(
    image,
    tf.tensor(boxes),
    tf.tensor([0], [1], 'int32'),
    cropSize,
  );
}

Box scaleBoxCoordinates(Box box, List<double> factor) {
  final startPoint = [
    box.startPoint[0] * factor[0],
    box.startPoint[1] * factor[1]
  ];
  final endPoint = [box.endPoint[0] * factor[0], box.endPoint[1] * factor[1]];
  final palmLandmarks = box.palmLandmarks!.map((coord) {
    final scaledCoord = [coord[0] * factor[0], coord[1] * factor[1]];
    return scaledCoord;
  }).toList();

  return Box(
      startPoint: startPoint, endPoint: endPoint, palmLandmarks: palmLandmarks);
}

Box enlargeBox(Box box, [num factor = 1.5]) {
  final center = getBoxCenter(box);
  final size = getBoxSize(box);

  final newHalfSize = [factor * size[0] / 2, factor * size[1] / 2];
  final startPoint = [center[0] - newHalfSize[0], center[1] - newHalfSize[1]];
  final endPoint = [center[0] + newHalfSize[0], center[1] + newHalfSize[1]];

  return Box(
    startPoint: startPoint,
    endPoint: endPoint,
    palmLandmarks: box.palmLandmarks,
  );
}

Box squarifyBox(Box box) {
  final centers = getBoxCenter(box);
  final size = getBoxSize(box);
  final maxEdge = size.reduce(Math.max);

  final halfSize = maxEdge / 2;
  final startPoint = [centers[0] - halfSize, centers[1] - halfSize];
  final endPoint = [centers[0] + halfSize, centers[1] + halfSize];

  return Box(
    startPoint: startPoint,
    endPoint: endPoint,
    palmLandmarks: box.palmLandmarks,
  );
}

Box shiftBox(Box box, List<double> shiftFactor) {
  final boxSize = [
    box.endPoint[0] - box.startPoint[0],
    box.endPoint[1] - box.startPoint[1]
  ];
  final shiftVector = [
    boxSize[0] * shiftFactor[0],
    boxSize[1] * shiftFactor[1]
  ];
  final startPoint = [
    box.startPoint[0] + shiftVector[0],
    box.startPoint[1] + shiftVector[1]
  ];
  final endPoint = [
    box.endPoint[0] + shiftVector[0],
    box.endPoint[1] + shiftVector[1]
  ];

  return Box(
      startPoint: startPoint,
      endPoint: endPoint,
      palmLandmarks: box.palmLandmarks);
}
