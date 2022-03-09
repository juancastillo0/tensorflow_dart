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
// import {TensorsToDetectionsConfig} from './interfaces/config_interfaces';
// import {AnchorTensor, Detection} from './interfaces/shape_interfaces';

import 'dart:typed_data';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'interfaces/common_interfaces.dart';
import 'interfaces/config_interfaces.dart';
import 'interfaces/shape_interfaces.dart';

/**
 * Convert result Tensors from object detection models into Detection boxes.
 *
 * @param detectionTensors List of Tensors of type Float32. The list of tensors
 *     can have 2 or 3 tensors. First tensor is the predicted raw
 *     boxes/keypoints. The size of the values must be
 *     (num_boxes * num_predicted_values). Second tensor is the score tensor.
 *     The size of the valuse must be (num_boxes * num_classes). It's optional
 *     to pass in a third tensor for anchors (e.g. for SSD models) depend on the
 *     outputs of the detection model. The size of anchor tensor must be
 *     (num_boxes * 4).
 * @param anchor A tensor for anchors. The size of anchor tensor must be
 *     (num_boxes * 4).
 * @param config
 */
Future<List<Detection>> tensorsToDetections(
  List<tf.Tensor> detectionTensors, // : [tf.Tensor1D, tf.Tensor2D]
  AnchorTensor anchor,
  TensorsToDetectionsConfig config,
) async {
  final rawScoreTensor = detectionTensors[0];
  final rawBoxTensor = detectionTensors[1];

  // Shape [numOfBoxes, 4] or [numOfBoxes, 12].
  final boxes = _decodeBoxes(rawBoxTensor, anchor, config);

  // Filter classes by scores.
  final normalizedScore = tf.tidy(() {
    var normalizedScore = rawScoreTensor;
    if (config.sigmoidScore == true) {
      if (config.scoreClippingThresh != null) {
        normalizedScore = tf.clipByValue(rawScoreTensor,
            -config.scoreClippingThresh!, config.scoreClippingThresh!);
      }
      normalizedScore = tf.sigmoid(normalizedScore);
      return normalizedScore;
    }

    return normalizedScore;
  });

  final outputDetections =
      await convertToDetections(boxes, normalizedScore, config);

  tf.dispose([boxes, normalizedScore]);

  return outputDetections;
}

Future<List<Detection>> convertToDetections(tf.Tensor2D detectionBoxes,
    tf.Tensor1D detectionScore, TensorsToDetectionsConfig config) async {
  final outputDetections = <Detection>[];
  final detectionBoxesData = await detectionBoxes.data() as Float32List;
  final detectionScoresData = await detectionScore.data() as Float32List;

  for (int i = 0; i < config.numBoxes; ++i) {
    if (config.minScoreThresh != null &&
        detectionScoresData[i] < config.minScoreThresh!) {
      continue;
    }
    final boxOffset = i * config.numCoords;
    final detection = _convertToDetection(
        detectionBoxesData[boxOffset + 0] /* boxYMin */,
        detectionBoxesData[boxOffset + 1] /* boxXMin */,
        detectionBoxesData[boxOffset + 2] /* boxYMax */,
        detectionBoxesData[boxOffset + 3] /* boxXMax */,
        detectionScoresData[i],
        config.flipVertically ?? false,
        i);
    final bbox = detection.locationData.relativeBoundingBox;

    if (bbox.width < 0 || bbox.height < 0) {
      // Decoded detection boxes could have negative values for width/height
      // due to model prediction. Filter out those boxes since some
      // downstream calculators may assume non-negative values.
      continue;
    }
    // Add keypoints.
    final numKeypoints = config.numKeypoints;
    if (numKeypoints != null && numKeypoints > 0) {
      final locationData = detection.locationData;
      final totalIdx = numKeypoints * config.numValuesPerKeypoint!;
      for (int kpId = 0;
          kpId < totalIdx;
          kpId += config.numValuesPerKeypoint!) {
        final keypointIndex = boxOffset + config.keypointCoordOffset! + kpId;
        final keypoint = Keypoint(
            x: detectionBoxesData[keypointIndex + 0],
            y: config.flipVertically == true
                ? 1 - detectionBoxesData[keypointIndex + 1]
                : detectionBoxesData[keypointIndex + 1]);
        locationData.relativeKeypoints.add(keypoint);
      }
    }
    outputDetections.add(detection);
  }

  return outputDetections;
}

Detection _convertToDetection(double boxYMin, double boxXMin, double boxYMax,
    double boxXMax, double score, bool flipVertically, int i) {
  return Detection(
    score: [score],
    ind: i,
    locationData: LocationData(
      relativeBoundingBox: BoundingBox(
        xMin: boxXMin,
        yMin: flipVertically ? 1 - boxYMax : boxYMin,
        xMax: boxXMax,
        yMax: flipVertically ? 1 - boxYMin : boxYMax,
        width: boxXMax - boxXMin,
        height: boxYMax - boxYMin,
      ),
    ),
  );
}

//[xCenter, yCenter, w, h, kp1, kp2, kp3, kp4]
//[yMin, xMin, yMax, xMax, kpX, kpY, kpX, kpY]
tf.Tensor2D _decodeBoxes(tf.Tensor2D rawBoxes, AnchorTensor anchor,
    TensorsToDetectionsConfig config) {
  return tf.tidy(() {
    late tf.Tensor yCenter;
    late tf.Tensor xCenter;
    late tf.Tensor h;
    late tf.Tensor w;
    final boxCoordOffset = config.boxCoordOffset;

    if (config.reverseOutputOrder == true) {
      // Shape [numOfBoxes, 1].
      xCenter =
          tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset! + 0], [-1, 1]));
      yCenter =
          tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset + 1], [-1, 1]));
      w = tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset + 2], [-1, 1]));
      h = tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset + 3], [-1, 1]));
    } else {
      yCenter =
          tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset! + 0], [-1, 1]));
      xCenter =
          tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset + 1], [-1, 1]));
      h = tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset + 2], [-1, 1]));
      w = tf.squeeze(tf.slice(rawBoxes, [0, boxCoordOffset + 3], [-1, 1]));
    }

    final xScale = tf.scalar(config.xScale!);
    final yScale = tf.scalar(config.yScale!);

    xCenter = tf.add(tf.mul(tf.div(xCenter, xScale), anchor.w), anchor.x);
    yCenter = tf.add(tf.mul(tf.div(yCenter, yScale), anchor.h), anchor.y);

    final hScale = tf.scalar(config.hScale!);
    final wScale = tf.scalar(config.wScale!);

    if (config.applyExponentialOnBoxSize == true) {
      h = tf.mul(tf.exp(tf.div(h, hScale)), anchor.h);
      w = tf.mul(tf.exp(tf.div(w, wScale)), anchor.w);
    } else {
      h = tf.mul(tf.div(h, hScale), anchor.h);
      w = tf.mul(tf.div(w, wScale), anchor.h);
    }
    final two = tf.scalar(2.0);

    final yMin = tf.sub(yCenter, tf.div(h, two));
    final xMin = tf.sub(xCenter, tf.div(w, two));
    final yMax = tf.add(yCenter, tf.div(h, two));
    final xMax = tf.add(xCenter, tf.div(w, two));

    // Shape [numOfBoxes, 4].
    var boxes = tf.concat([
      tf.reshape(yMin, [config.numBoxes, 1]),
      tf.reshape(xMin, [config.numBoxes, 1]),
      tf.reshape(yMax, [config.numBoxes, 1]),
      tf.reshape(xMax, [config.numBoxes, 1])
    ], 1);

    if (config.numKeypoints != null) {
      for (int k = 0; k < config.numKeypoints!; ++k) {
        final keypointOffset =
            config.keypointCoordOffset! + k * config.numValuesPerKeypoint!;
        tf.Tensor keypointX;
        tf.Tensor keypointY;
        if (config.reverseOutputOrder == true) {
          keypointX =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset], [-1, 1]));
          keypointY =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset + 1], [-1, 1]));
        } else {
          keypointY =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset], [-1, 1]));
          keypointX =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset + 1], [-1, 1]));
        }
        final keypointXNormalized =
            tf.add(tf.mul(tf.div(keypointX, xScale), anchor.w), anchor.x);
        final keypointYNormalized =
            tf.add(tf.mul(tf.div(keypointY, yScale), anchor.h), anchor.y);
        boxes = tf.concat([
          boxes,
          tf.reshape(keypointXNormalized, [config.numBoxes, 1]),
          tf.reshape(keypointYNormalized, [config.numBoxes, 1])
        ], 1);
      }
    }

    // Shape [numOfBoxes, 4] || [numOfBoxes, 12].
    return boxes as tf.Tensor2D;
  });
}
