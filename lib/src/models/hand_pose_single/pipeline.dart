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

// import {Box, cutBoxFromImageAndResize, enlargeBox, getBoxCenter, getBoxSize, shiftBox, squarifyBox} from './box';
// import {HandDetector} from './hand';
// import {buildRotationMatrix, computeRotation, dot, invertTransformMatrix, rotatePoint, TransformationMatrix} from './util';

import 'dart:math' as Math;
import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;

import 'box.dart';
import 'hand.dart';
import 'util.dart';

const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.8;

const PALM_BOX_SHIFT_VECTOR = [0.0, -0.4];
const PALM_BOX_ENLARGE_FACTOR = 3;

const HAND_BOX_SHIFT_VECTOR = [0.0, -0.1];
const HAND_BOX_ENLARGE_FACTOR = 1.65;

const PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2];
const PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0;
const PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2;

typedef Coords3D = List<List<double>>; // Array<[number, number, number]>;
typedef Coords2D = List<List<double>>; // Array<[number, number]>;

class Prediction {
  final double handInViewConfidence;
  final Coords3D landmarks;
  final BoundingBox boundingBox;

  const Prediction({
    required this.handInViewConfidence,
    required this.landmarks,
    required this.boundingBox,
  });
}

class BoundingBox {
  final List<double> topLeft;
  final List<double> bottomRight;

  const BoundingBox({
    required this.topLeft,
    required this.bottomRight,
  });
}

// The Pipeline coordinates between the bounding box and skeleton models.
class HandPipeline {
  // TODO(annxingyuan): Add multi-hand support.
  int get maxHandsNumber => 1;

  // An array of hand bounding boxes.
  List<Box> regionsOfInterest = [];
  int runsWithoutHandDetector = 0;
  final HandDetector boundingBoxDetector;
  /* MediaPipe model for detecting hand bounding box */
  final tfconv.GraphModel meshDetector;
  /* MediaPipe model for detecting hand mesh */
  final int meshWidth;
  final int meshHeight;
  final double maxContinuousChecks;
  final double detectionConfidence;

  HandPipeline(
    this.boundingBoxDetector,
    /* MediaPipe model for detecting hand bounding box */
    this.meshDetector,
    /* MediaPipe model for detecting hand mesh */
    this.meshWidth,
    this.meshHeight,
    this.maxContinuousChecks,
    this.detectionConfidence,
  );

  // Get the bounding box surrounding the hand, given palm landmarks.
  Box _getBoxForPalmLandmarks(
      Coords2D palmLandmarks, TransformationMatrix rotationMatrix) {
    final Coords2D rotatedPalmLandmarks = palmLandmarks.map((coord) {
      final homogeneousCoordinate = [...coord, 1.0];
      return rotatePoint(homogeneousCoordinate, rotationMatrix);
    }).toList();

    final boxAroundPalm =
        this._calculateLandmarksBoundingBox(rotatedPalmLandmarks);
    // boxAroundPalm only surrounds the palm - therefore we shift it
    // upwards so it will capture fingers once enlarged + squarified.
    return enlargeBox(
        squarifyBox(shiftBox(boxAroundPalm, PALM_BOX_SHIFT_VECTOR)),
        PALM_BOX_ENLARGE_FACTOR);
  }

  // Get the bounding box surrounding the hand, given all hand landmarks.
  Box _getBoxForHandLandmarks(Coords3D landmarks) {
    // The MediaPipe hand mesh model is trained on hands with empty space
    // around them, so we still need to shift / enlarge boxAroundHand even
    // though it surrounds the entire hand.
    final boundingBox = this._calculateLandmarksBoundingBox(landmarks);
    final Box boxAroundHand = enlargeBox(
      squarifyBox(shiftBox(boundingBox, HAND_BOX_SHIFT_VECTOR)),
      HAND_BOX_ENLARGE_FACTOR,
    );

    final Coords2D palmLandmarks = [];
    for (int i = 0; i < PALM_LANDMARK_IDS.length; i++) {
      palmLandmarks.add(landmarks[PALM_LANDMARK_IDS[i]].slice(0, 2));
    }

    return Box(
      endPoint: boxAroundHand.endPoint,
      startPoint: boxAroundHand.startPoint,
      palmLandmarks: palmLandmarks,
    );
  }

  // Scale, rotate, and translate raw keypoints from the model so they map to
  // the input coordinates.
  Coords3D _transformRawCoords(Coords3D rawCoords, Box box, double angle,
      TransformationMatrix rotationMatrix) {
    final boxSize = getBoxSize(box);
    final scaleFactor = [
      boxSize[0] / this.meshWidth,
      boxSize[1] / this.meshHeight
    ];

    final coordsScaled = rawCoords.map((coord) {
      return [
        scaleFactor[0] * (coord[0] - this.meshWidth / 2),
        scaleFactor[1] * (coord[1] - this.meshHeight / 2),
        coord[2]
      ];
    });

    final coordsRotationMatrix = buildRotationMatrix(angle, [0, 0]);
    final coordsRotated = coordsScaled.map((coord) {
      final rotated = rotatePoint(coord, coordsRotationMatrix);
      return [...rotated, coord[2]];
    });

    final inverseRotationMatrix = invertTransformMatrix(rotationMatrix);
    final boxCenter = [...getBoxCenter(box), 1.0];

    final originalBoxCenter = [
      dot(boxCenter, inverseRotationMatrix[0]),
      dot(boxCenter, inverseRotationMatrix[1])
    ];

    return coordsRotated.map((coord) {
      return [
        coord[0] + originalBoxCenter[0],
        coord[1] + originalBoxCenter[1],
        coord[2]
      ];
    }).toList();
  }

  Future<Prediction?> estimateHand(tf.Tensor4D image) async {
    final useFreshBox = this._shouldUpdateRegionsOfInterest();
    if (useFreshBox == true) {
      final boundingBoxPrediction =
          await this.boundingBoxDetector.estimateHandBounds(image);
      if (boundingBoxPrediction == null) {
        image.dispose();
        this.regionsOfInterest = [];
        return null;
      }

      this._updateRegionsOfInterest(
          boundingBoxPrediction, true /*force update*/);
      this.runsWithoutHandDetector = 0;
    } else {
      this.runsWithoutHandDetector++;
    }

    // Rotate input so the hand is vertically oriented.
    final currentBox = this.regionsOfInterest[0];
    final angle = computeRotation(
      currentBox.palmLandmarks![PALM_LANDMARKS_INDEX_OF_PALM_BASE],
      currentBox.palmLandmarks![PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE],
    );

    final palmCenter = getBoxCenter(currentBox);
    final palmCenterNormalized = [
      palmCenter[0] / image.shape[2],
      palmCenter[1] / image.shape[1]
    ];
    final rotatedImage =
        tf.image.rotateWithOffset(image, angle, center: palmCenterNormalized);

    final rotationMatrix = buildRotationMatrix(-angle, palmCenter);

    final Box box;
    // The bounding box detector only detects palms, so if we're using a fresh
    // bounding box prediction, we have to construct the hand bounding box from
    // the palm keypoints.
    if (useFreshBox == true) {
      box = this._getBoxForPalmLandmarks(
        currentBox.palmLandmarks!,
        rotationMatrix,
      );
    } else {
      box = currentBox;
    }

    final croppedInput = cutBoxFromImageAndResize(
        box, rotatedImage, [this.meshWidth, this.meshHeight]);
    final handImage = tf.div(croppedInput, tf.scalar(255.0));
    croppedInput.dispose();
    rotatedImage.dispose();

    final List<tf.Tensor> prediction;
    if (tf.getBackend() == 'webgl') {
      // Currently tfjs-core does not pack depthwiseConv because it fails for
      // very large inputs (https://github.com/tensorflow/tfjs/issues/1652).
      // TODO(annxingyuan): call tf.enablePackedDepthwiseConv when available
      // (https://github.com/tensorflow/tfjs/issues/2821)
      final savedWebglPackDepthwiseConvFlag =
          tf.env().get('WEBGL_PACK_DEPTHWISECONV');
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
      prediction = this.meshDetector.predict(handImage) as List<tf.Tensor>;
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
    } else {
      prediction = this.meshDetector.predict(handImage) as List<tf.Tensor>;
    }

    final flag = prediction[0];
    final keypoints = prediction[1];

    handImage.dispose();

    final flagValue = flag.dataSync()[0];
    flag.dispose();

    if (flagValue < this.detectionConfidence) {
      keypoints.dispose();
      this.regionsOfInterest = [];
      return null;
    }

    final keypointsReshaped = tf.reshape(keypoints, [-1, 3]);
    // Calling arraySync() because the tensor is very small so it's not worth
    // calling await array().
    final rawCoords = keypointsReshaped.arraySync() as Coords3D;
    keypoints.dispose();
    keypointsReshaped.dispose();

    final coords =
        this._transformRawCoords(rawCoords, box, angle, rotationMatrix);
    final nextBoundingBox = this._getBoxForHandLandmarks(coords);

    this._updateRegionsOfInterest(nextBoundingBox, false /* force replace */);

    final result = Prediction(
      landmarks: coords,
      handInViewConfidence: flagValue,
      boundingBox: BoundingBox(
        topLeft: nextBoundingBox.startPoint,
        bottomRight: nextBoundingBox.endPoint,
      ),
    );

    return result;
  }

  Box _calculateLandmarksBoundingBox(List<List<double>> landmarks) {
    final xs = landmarks.map((d) => d[0]);
    final ys = landmarks.map((d) => d[1]);
    final startPoint = [xs.reduce(Math.min), ys.reduce(Math.min)];
    final endPoint = [xs.reduce(Math.max), ys.reduce(Math.max)];
    return Box(
      startPoint: startPoint,
      endPoint: endPoint,
    );
  }

  // Updates regions of interest if the intersection over union between
  // the incoming and previous regions falls below a threshold.
  void _updateRegionsOfInterest(Box box, bool forceUpdate) {
    if (forceUpdate) {
      this.regionsOfInterest = [box];
    } else {
      final previousBox = this.regionsOfInterest[0];
      double iou = 0;

      if (previousBox != null && previousBox.startPoint != null) {
        final boxStartX = box.startPoint[0];
        final boxStartY = box.startPoint[1];
        final boxEndX = box.endPoint[0];
        final boxEndY = box.endPoint[1];
        final previousBoxStartX = previousBox.startPoint[0];
        final previousBoxStartY = previousBox.startPoint[1];
        final previousBoxEndX = previousBox.endPoint[0];
        final previousBoxEndY = previousBox.endPoint[1];

        final xStartMax = Math.max(boxStartX, previousBoxStartX);
        final yStartMax = Math.max(boxStartY, previousBoxStartY);
        final xEndMin = Math.min(boxEndX, previousBoxEndX);
        final yEndMin = Math.min(boxEndY, previousBoxEndY);

        final intersection = (xEndMin - xStartMax) * (yEndMin - yStartMax);
        final boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
        final previousBoxArea = (previousBoxEndX - previousBoxStartX) *
            (previousBoxEndY - boxStartY);
        iou = intersection / (boxArea + previousBoxArea - intersection);
      }

      this.regionsOfInterest[0] =
          iou > UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD ? previousBox : box;
    }
  }

  bool _shouldUpdateRegionsOfInterest() {
    final roisCount = this.regionsOfInterest.length;

    return roisCount != this.maxHandsNumber ||
        this.runsWithoutHandDetector >= this.maxContinuousChecks;
  }
}
