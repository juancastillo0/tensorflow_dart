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

// import {HandDetector} from './hand';
// import {MESH_ANNOTATIONS} from './keypoints';
// import {Coords3D, HandPipeline, Prediction} from './pipeline';

import 'dart:convert';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;
import 'package:http/http.dart' as http;

import 'hand.dart' show AnchorsConfig, HandDetector;
import 'keypoints.dart' show MESH_ANNOTATIONS;
import 'pipeline.dart';

// Load the bounding box detector model.
Future<tfconv.GraphModel> _loadHandDetectorModel() async {
  const HANDDETECT_MODEL_PATH =
      'https://tfhub.dev/mediapipe/tfjs-model/handdetector/1/default/1';
  return tfconv.loadGraphModel(
    tfconv.ModelHandler.fromUrl(HANDDETECT_MODEL_PATH),
    tfconv.LoadOptions(fromTFHub: true),
  );
}

const MESH_MODEL_INPUT_WIDTH = 256;
const MESH_MODEL_INPUT_HEIGHT = 256;

// Load the mesh detector model.
Future<tfconv.GraphModel> _loadHandPoseModel() async {
  const HANDPOSE_MODEL_PATH =
      'https://tfhub.dev/mediapipe/tfjs-model/handskeleton/1/default/1';
  return tfconv.loadGraphModel(
    tfconv.ModelHandler.fromUrl(HANDPOSE_MODEL_PATH),
    tfconv.LoadOptions(fromTFHub: true),
  );
}

// In single shot detector pipelines, the output space is discretized into a set
// of bounding boxes, each of which is assigned a score during prediction. The
// anchors define the coordinates of these boxes.
Future<List<AnchorsConfig>> _loadAnchors() async {
  final _stream = await tf.env().platform!.fetch(
      Uri.parse(
          'https://tfhub.dev/mediapipe/tfjs-model/handskeleton/1/default/1/anchors.json?tfjs-format=file'),
      null);

  final response = await http.Response.fromStream(_stream);
  return (jsonDecode(response.body) as List)
      .map((e) => AnchorsConfig.fromJson(e))
      .toList();
}

class AnnotatedPrediction implements Prediction {
  final Map<String, List<List<double>>> annotations;
  final double handInViewConfidence;
  final Coords3D landmarks;
  final BoundingBox boundingBox;

  const AnnotatedPrediction({
    required this.annotations,
    required this.handInViewConfidence,
    required this.landmarks,
    required this.boundingBox,
  });
}

/**
 * Load handpose.
 *
 * @param config A configuration object with the following properties:
 * - `maxContinuousChecks` How many frames to go without running the bounding
 * box detector. Defaults to infinity. Set to a lower value if you want a safety
 * net in case the mesh detector produces consistently flawed predictions.
 * - `detectionConfidence` Threshold for discarding a prediction. Defaults to
 * 0.8.
 * - `iouThreshold` A float representing the threshold for deciding whether
 * boxes overlap too much in non-maximum suppression. Must be between [0, 1].
 * Defaults to 0.3.
 * - `scoreThreshold` A threshold for deciding when to remove boxes based
 * on score in non-maximum suppression. Defaults to 0.75.
 */
Future<HandPose> load({
  double maxContinuousChecks = double.infinity,
  double detectionConfidence = 0.8,
  double iouThreshold = 0.3,
  double scoreThreshold = 0.5,
}) async {
  final _models = await Future.wait(
      [_loadAnchors(), _loadHandDetectorModel(), _loadHandPoseModel()]);
  final ANCHORS = _models[0] as List<AnchorsConfig>;
  final handDetectorModel = _models[1] as tfconv.GraphModel;
  final handPoseModel = _models[2] as tfconv.GraphModel;

  final detector = HandDetector(handDetectorModel, MESH_MODEL_INPUT_WIDTH,
      MESH_MODEL_INPUT_HEIGHT, ANCHORS, iouThreshold, scoreThreshold);
  final pipeline = HandPipeline(detector, handPoseModel, MESH_MODEL_INPUT_WIDTH,
      MESH_MODEL_INPUT_HEIGHT, maxContinuousChecks, detectionConfidence);
  final handpose = HandPose(pipeline);

  return handpose;
}

List<int> _getInputTensorDimensions(Object input
// : tf.Tensor3D|ImageData|HTMLVideoElement|
//                                   HTMLImageElement|HTMLCanvasElement
    ) {
  return input is tf.Tensor
      ? [input.shape[0], input.shape[1]]
      : [(input as dynamic).height, (input as dynamic).width];
}

Prediction _flipHandHorizontal(Prediction prediction, int width) {
  final handInViewConfidence = prediction.handInViewConfidence;
  final landmarks = prediction.landmarks;
  final boundingBox = prediction.boundingBox;

  return Prediction(
    handInViewConfidence: handInViewConfidence,
    landmarks: landmarks.map((coord) {
      return [width - 1 - coord[0], coord[1], coord[2]];
    }).toList(),
    boundingBox: BoundingBox(topLeft: [
      width - 1 - boundingBox.topLeft[0],
      boundingBox.topLeft[1]
    ], bottomRight: [
      width - 1 - boundingBox.bottomRight[0],
      boundingBox.bottomRight[1]
    ]),
  );
}

class HandPose {
  final HandPipeline pipeline;
  HandPose(this.pipeline);

  static Map<String, List<int>> getAnnotations() {
    return MESH_ANNOTATIONS;
  }

  /**
   * Finds hands in the input image.
   *
   * @param input The image to classify. Can be a tensor, DOM element image,
   * video, or canvas.
   * @param flipHorizontal Whether to flip the hand keypoints horizontally.
   * Should be true for videos that are flipped by default (e.g. webcams).
   */
  Future<List<AnnotatedPrediction>> estimateHands(
    // : tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement| HTMLCanvasElement,
    Object input, {
    bool flipHorizontal = false,
  }) async {
    final width = _getInputTensorDimensions(input)[1];

    final tf.Tensor4D image = tf.tidy(() {
      if (input is! tf.Tensor) {
        input = tf.browser.fromPixels(input);
      }
      return tf.expandDims(tf.cast(input as tf.Tensor, 'float32'));
    });

    final result = await this.pipeline.estimateHand(image);
    image.dispose();

    if (result == null) {
      return [];
    }

    var prediction = result;
    if (flipHorizontal == true) {
      prediction = _flipHandHorizontal(result, width);
    }

    final Map<String, Coords3D> annotations = {};
    for (final key in MESH_ANNOTATIONS.keys) {
      annotations[key] = MESH_ANNOTATIONS[key]!
          .map((index) => prediction.landmarks[index])
          .toList();
    }

    return [
      AnnotatedPrediction(
        handInViewConfidence: prediction.handInViewConfidence,
        boundingBox: prediction.boundingBox,
        landmarks: prediction.landmarks,
        annotations: annotations,
      ),
    ];
  }
}
