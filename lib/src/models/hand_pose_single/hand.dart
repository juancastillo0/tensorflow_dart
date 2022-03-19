import 'dart:convert';

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

// import * as tfconv from '@tensorflow/tfjs-converter';
// import * as tf from '@tensorflow/tfjs-core';

// import {Box, scaleBoxCoordinates} from './box';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;

import 'box.dart';

class HandDetectorPrediction {
  final tf.Tensor2D boxes;
  final tf.Tensor2D palmLandmarks;

  const HandDetectorPrediction({
    required this.boxes,
    required this.palmLandmarks,
  });
}

class AnchorsConfig {
  final double w;
  final double h;
  final double x_center;
  final double y_center;

  const AnchorsConfig({
    required this.w,
    required this.h,
    required this.x_center,
    required this.y_center,
  });

  Map<String, dynamic> toJson() {
    return {
      'w': w,
      'h': h,
      'x_center': x_center,
      'y_center': y_center,
    };
  }

  factory AnchorsConfig.fromJson(Map<String, dynamic> map) {
    return AnchorsConfig(
      w: (map['w'] as num).toDouble(),
      h: (map['h'] as num).toDouble(),
      x_center: (map['x_center'] as num).toDouble(),
      y_center: (map['y_center'] as num).toDouble(),
    );
  }
}

class HandDetector {
  late final List<List<double>> anchors;
  late final tf.Tensor2D anchorsTensor;
  late final tf.Tensor1D inputSizeTensor;
  late final tf.Tensor1D doubleInputSizeTensor;

  final tfconv.GraphModel model;
  final int width;
  final int height;
  final List<AnchorsConfig> anchorsAnnotated;
  final double iouThreshold;
  final double scoreThreshold;

  HandDetector(
    this.model,
    this.width,
    this.height,
    this.anchorsAnnotated,
    this.iouThreshold,
    this.scoreThreshold,
  ) {
    this.anchors = anchorsAnnotated
        .map((anchor) => ([anchor.x_center, anchor.y_center]))
        .toList();
    this.anchorsTensor = tf.tensor2d(this.anchors);
    this.inputSizeTensor = tf.tensor1d([width, height]);
    this.doubleInputSizeTensor = tf.tensor1d([width * 2, height * 2]);
  }

  tf.Tensor2D _normalizeBoxes(tf.Tensor2D boxes) {
    return tf.tidy(() {
      final boxOffsets = tf.slice(boxes, [0, 0], [-1, 2]);
      final boxSizes = tf.slice(boxes, [0, 2], [-1, 2]);

      final boxCenterPoints =
          tf.add(tf.div(boxOffsets, this.inputSizeTensor), this.anchorsTensor);
      final halfBoxSizes = tf.div(boxSizes, this.doubleInputSizeTensor);

      final tf.Tensor2D startPoints =
          tf.mul(tf.sub(boxCenterPoints, halfBoxSizes), this.inputSizeTensor);
      final tf.Tensor2D endPoints =
          tf.mul(tf.add(boxCenterPoints, halfBoxSizes), this.inputSizeTensor);
      return tf.concat([startPoints, endPoints], 1);
    });
  }

  tf.Tensor2D _normalizeLandmarks(tf.Tensor2D rawPalmLandmarks, int index) {
    return tf.tidy(() {
      final landmarks = tf.add(
        tf.div(tf.reshape(rawPalmLandmarks, [-1, 7, 2]), this.inputSizeTensor),
        tf.tensor(this.anchors[index]),
      );

      return tf.mul(landmarks, this.inputSizeTensor);
    });
  }

  Future<HandDetectorPrediction?> _getBoundingBoxes(tf.Tensor4D input) async {
    final normalizedInput = tf.tidy(
      () => tf.mul(tf.sub(input, tf.scalar(0.5)), tf.scalar(2.0)),
    );

    final tf.Tensor3D batchedPrediction;
    if (tf.getBackend() == 'webgl') {
      // Currently tfjs-core does not pack depthwiseConv because it fails for
      // very large inputs (https://github.com/tensorflow/tfjs/issues/1652).
      // TODO(annxingyuan): call tf.enablePackedDepthwiseConv when available
      // (https://github.com/tensorflow/tfjs/issues/2821)
      final savedWebglPackDepthwiseConvFlag =
          tf.env().get('WEBGL_PACK_DEPTHWISECONV');
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
      // The model returns a tensor with the following shape:
      //  [1 (batch), 2944 (anchor points), 19 (data for each anchor)]
      batchedPrediction = this.model.predict(normalizedInput) as tf.Tensor3D;
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
    } else {
      batchedPrediction = this.model.predict(normalizedInput) as tf.Tensor3D;
    }

    final tf.Tensor2D prediction = tf.squeeze(batchedPrediction);

    // Regression score for each anchor point.
    final tf.Tensor1D scores = tf.tidy(
        () => tf.squeeze(tf.sigmoid(tf.slice(prediction, [0, 0], [-1, 1]))));

    // Bounding box for each anchor point.
    final rawBoxes = tf.slice(prediction, [0, 1], [-1, 4]);
    final boxes = this._normalizeBoxes(rawBoxes);

    final boxesWithHandsTensor = tf.image.nonMaxSuppression(boxes, scores, 1,
        iouThreshold: this.iouThreshold, scoreThreshold: this.scoreThreshold);
    final boxesWithHands = await boxesWithHandsTensor.array() as List;

    final toDispose = [
      normalizedInput,
      batchedPrediction,
      boxesWithHandsTensor,
      prediction,
      boxes,
      rawBoxes,
      scores
    ];
    if (boxesWithHands.length == 0) {
      toDispose.forEach((tensor) => tensor.dispose());

      return null;
    }

    final boxIndex = boxesWithHands[0];
    final matchingBox = tf.slice(boxes, [boxIndex, 0], [1, -1]);

    final rawPalmLandmarks = tf.slice(prediction, [boxIndex, 5], [1, 14]);
    final tf.Tensor2D palmLandmarks = tf.tidy(() => tf.reshape(
        this._normalizeLandmarks(rawPalmLandmarks, boxIndex), [-1, 2]));

    toDispose.add(rawPalmLandmarks);
    toDispose.forEach((tensor) => tensor.dispose());

    return HandDetectorPrediction(
        boxes: matchingBox, palmLandmarks: palmLandmarks);
  }

  /**
   * Returns a Box identifying the bounding box of a hand within the image.
   * Returns null if there is no hand in the image.
   *
   * @param input The image to classify.
   */
  Future<Box?> estimateHandBounds(tf.Tensor4D input) async {
    final inputHeight = input.shape[1];
    final inputWidth = input.shape[2];

    final tf.Tensor4D image = tf.tidy(() => tf.div(
        tf.image.resizeBilinear(input, [this.width, this.height]),
        tf.scalar(255.0)));
    final prediction = await this._getBoundingBoxes(image);

    if (prediction == null) {
      image.dispose();
      return null;
    }

    // Calling arraySync on both boxes and palmLandmarks because the tensors are
    // very small so it's not worth calling await array().
    final boundingBoxes = (prediction.boxes.arraySync() as List)
        .map((e) => (e as List).cast<double>())
        .toList();
    final startPoint = boundingBoxes[0].slice(0, 2);
    final endPoint = boundingBoxes[0].slice(2, 4);
    final palmLandmarks = (prediction.palmLandmarks.arraySync() as List)
        .map((e) => (e as List).cast<double>())
        .toList();

    image.dispose();
    prediction.boxes.dispose();
    prediction.palmLandmarks.dispose();

    return scaleBoxCoordinates(
      Box(
        startPoint: startPoint,
        endPoint: endPoint,
        palmLandmarks: palmLandmarks,
      ),
      [inputWidth / this.width, inputHeight / this.height],
    );
  }
}
