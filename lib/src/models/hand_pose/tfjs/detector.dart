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

// import * as tfconv from '@tensorflow/tfjs-converter';
// import * as tf from '@tensorflow/tfjs-core';

// import {MEDIAPIPE_KEYPOINTS} from '../constants';
// import {HandDetector} from '../hand_detector';
// import {MediaPipeHandsTfjsEstimationConfig, MediaPipeHandsTfjsModelConfig} from './types';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;

import '../../shared/association_norm_rect.dart';
import '../../shared/calculate_landmark_projection.dart';
import '../../shared/calculate_world_landmark_projection.dart';
import '../../shared/convert_image_to_tensor.dart';
import '../../shared/create_ssd_anchors.dart';
import '../../shared/detection_to_rect.dart';
import '../../shared/detector_result.dart';
import '../../shared/image_utils.dart';
import '../../shared/interfaces/common_interfaces.dart';
import '../../shared/interfaces/config_interfaces.dart';
import '../../shared/interfaces/shape_interfaces.dart';
import '../../shared/non_max_suppression.dart';
import '../../shared/normalized_keypoints_to_keypoints.dart';
import '../../shared/remove_detection_letterbox.dart';
import '../../shared/remove_landmark_letterbox.dart';
import '../../shared/tensors_to_detections.dart';
import '../../shared/tensors_to_landmarks.dart';
import '../../shared/transform_rect.dart';
import '../constants.dart' show MEDIAPIPE_KEYPOINTS;
import '../hand_detector.dart' show HandDetector;
import '../types.dart' show Hand, HandDetectorInput, Handedness;
import 'calculators/hand_landmarks_to_rect.dart';
import 'detector_utils.dart';
import 'types.dart';
import 'constants.dart' as constants;

class HandLandmarksResult {
  final List<Keypoint> landmarks;
  final List<Keypoint> worldLandmarks;
  final double handScore;
  final Handedness handedness;

  const HandLandmarksResult({
    required this.landmarks,
    required this.worldLandmarks,
    required this.handScore,
    required this.handedness,
  });
}

/**
 * MediaPipeHands detector class.
 */
class MediaPipeHandsTfjsDetector implements HandDetector {
  // TODO: all private
  final List<Rect> anchors;
  late final AnchorTensor anchorTensor;

  // Store global states.
  List<Rect>? prevHandRectsFromLandmarks = null;

  final tfconv.GraphModel detectorModel;
  final tfconv.GraphModel landmarkModel;
  final int maxHands;

  MediaPipeHandsTfjsDetector(
      this.detectorModel, this.landmarkModel, this.maxHands)
      : anchors =
            createSsdAnchors(constants.MPHANDS_DETECTOR_ANCHOR_CONFIGURATION) {
    final anchorW = tf.tensor1d(this.anchors.map((a) => a.width).toList());
    final anchorH = tf.tensor1d(this.anchors.map((a) => a.height).toList());
    final anchorX = tf.tensor1d(this.anchors.map((a) => a.xCenter).toList());
    final anchorY = tf.tensor1d(this.anchors.map((a) => a.yCenter).toList());
    this.anchorTensor = AnchorTensor(
      x: anchorX,
      y: anchorY,
      w: anchorW,
      h: anchorH,
    );
  }

  /**
   * Estimates hands for an image or video frame.
   *
   * It returns a single hand or multiple hands based on the maxHands
   * parameter from the `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param estimationConfig Optional. See `MediaPipeHandsTfjsEstimationConfig`
   *       documentation for detail.
   *
   * @return An array of `Hand`s.
   */
  // TF.js implementation of the mediapipe hand detection pipeline.
  // ref graph:
  // https://github.com/google/mediapipe/blob/master/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt
  Future<List<Hand>> estimateHands(
    HandDetectorInput image,
    MediaPipeHandsTfjsEstimationConfig? estimationConfig,
  ) async {
    final config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      this.reset();
      return [];
    }

    // HandLandmarkTrackingCpu: ImagePropertiesCalculator
    // Extracts image size.
    final imageSize = getImageSize(image);

    final image3d = tf.tidy(() {
      var imageTensor = tf.cast(toImageTensor(image), 'float32');
      if (config.flipHorizontal == true) {
        final batchAxis = 0;
        imageTensor = tf.squeeze(tf.image.flipLeftRight(
            // tslint:disable-next-line: no-unnecessary-type-assertion
            tf.expandDims(imageTensor, batchAxis) as tf.Tensor4D), [batchAxis]);
      }
      return imageTensor;
    });

    final prevHandRectsFromLandmarks = this.prevHandRectsFromLandmarks;
    List<Rect> handRects;
    // Drops the incoming image for detection if enough hands have already been
    // identified from the previous image. Otherwise, passes the incoming image
    // through to trigger a new round of palm detection.
    if (config.staticImageMode == true ||
        prevHandRectsFromLandmarks == null ||
        prevHandRectsFromLandmarks.length < this.maxHands) {
      // HandLandmarkTrackingCpu: PalmDetectionCpu
      // Detects palms.
      final allPalmDetections = await this._detectPalm(image3d);

      if (allPalmDetections.length == 0) {
        this.reset();
        image3d.dispose();
        return [];
      }

      // HandLandmarkTrackingCpu: ClipDetectionVectorSizeCalculator
      // HandLandmarkTrackingCpu: Makes sure there are no more detections than
      // the provided maxHands. This is already done by our implementation of
      // nonMaxSuppresion.
      final palmDetections = allPalmDetections;

      // HandLandmarkTrackingCpu: PalmDetectionDetectionToRoi
      // Calculates region of interest (ROI) based on the specified palm.
      final handRectsFromPalmDetections = palmDetections
          .map((detection) => this._palmDetectionToRoi(detection, imageSize))
          .toList();

      handRects = handRectsFromPalmDetections;
    } else {
      handRects = prevHandRectsFromLandmarks;
    }

    // HandLandmarkTrackingCpu: AssociationNormRectCalculator
    // This calculator ensures that the output handRects array
    // doesn't contain overlapping regions based on the specified
    // minSimilarityThreshold. Note that our implementation does not perform
    // association between rects from previous image and rects based
    // on palm detections from the current image due to not having tracking
    // IDs in our API, so we don't call it with two inputs like MediaPipe
    // (previous and new rects). The call is nonetheless still necessary
    // since rects from previous image could overlap.
    handRects = calculateAssociationNormRect(
        [handRects], constants.MPHANDS_MIN_SIMILARITY_THRESHOLD);

    // HandLandmarkTrackingCpu: HandLandmarkCpu
    // Detect hand landmarks for the specific hand rect.
    final handResults = await Future.wait(
        handRects.map((handRect) => this._handLandmarks(handRect, image3d)));

    final hands = <Hand>[];
    this.prevHandRectsFromLandmarks = [];

    for (final handResult in handResults) {
      if (handResult == null) {
        continue;
      }

      final landmarks = handResult.landmarks;
      final worldLandmarks = handResult.worldLandmarks;
      final handedness = handResult.handedness;

      final score = handResult.handScore;

      // HandLandmarkTrackingCpu: HandLandmarkLandmarksToRoi
      // Calculate region of interest (ROI) based on detected hand landmarks to
      // reuse on the subsequent runs of the graph.
      this
          .prevHandRectsFromLandmarks!
          .add(this._handLandmarksToRoi(landmarks, imageSize));

      // Scale back keypoints.
      var keypoints = normalizedKeypointsToKeypoints(landmarks, imageSize);

      // Add keypoint name.
      if (keypoints != null) {
        keypoints = keypoints.mapIndexed((i, keypoint) {
          return keypoint.copyWith(
            name: Nullable(MEDIAPIPE_KEYPOINTS[i]),
            z: Nullable(null),
          );
        }).toList();
      }

      var keypoints3D = worldLandmarks;

      // Add keypoint name.
      if (keypoints3D != null) {
        keypoints3D = keypoints3D.mapIndexed((i, keypoint3D) {
          return keypoint3D.copyWith(
            name: Nullable(MEDIAPIPE_KEYPOINTS[i]),
          );
        }).toList();
      }

      hands.add(Hand(
        keypoints: keypoints,
        keypoints3D: keypoints3D,
        handedness: handedness,
        score: score,
      ));
    }
    image3d.dispose();

    return hands;
  }

  void dispose() {
    this.detectorModel.dispose();
    this.landmarkModel.dispose();
    tf.dispose([
      this.anchorTensor.x,
      this.anchorTensor.y,
      this.anchorTensor.w,
      this.anchorTensor.h
    ]);
  }

  void reset() {
    this.prevHandRectsFromLandmarks = null;
  }

  // Detects palms.
  // Subgraph: PalmDetectionCpu.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
  Future<List<Detection>> _detectPalm(HandDetectorInput image) async {
    // PalmDetectionCpu: ImageToTensorCalculator
    // Transforms the input image into a 128x128 while keeping the aspect ratio
    // resulting in potential letterboxing in the transformed image.
    final _imageTensor = convertImageToTensor(
      image,
      constants.MPHANDS_DETECTOR_IMAGE_TO_TENSOR_CONFIG,
      null,
    );

    final imageValueShifted = _imageTensor.imageTensor;
    final padding = _imageTensor.padding;

    final detectionResult =
        this.detectorModel.predict(imageValueShifted) as tf.Tensor3D;
    // PalmDetectionCpu: InferenceCalculator
    // The model returns a tensor with the following shape:
    // [1 (batch), 896 (anchor points), 19 (data for each anchor)]
    final _result = detectorResult(detectionResult);
    final boxes = _result.boxes;
    final logits = _result.logits;

    // PalmDetectionCpu: TensorsToDetectionsCalculator
    final List<Detection> detections = await tensorsToDetections(
        [logits, boxes],
        this.anchorTensor,
        constants.MPHANDS_TENSORS_TO_DETECTION_CONFIGURATION);

    if (detections.length == 0) {
      tf.dispose([imageValueShifted, detectionResult, logits, boxes]);
      return detections;
    }

    // PalmDetectionCpu: NonMaxSuppressionCalculator
    final selectedDetections = await nonMaxSuppression(
      detections, this.maxHands,
      constants.MPHANDS_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION
          .minSuppressionThreshold,
      // constants.MPHANDS_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION
      //     .overlapType,
    );

    // PalmDetectionCpu: DetectionLetterboxRemovalCalculator
    final newDetections = removeDetectionLetterbox(selectedDetections, padding);

    tf.dispose([imageValueShifted, detectionResult, logits, boxes]);

    return newDetections;
  }

  // calculates hand ROI from palm detection.
  // Subgraph: PalmDetectionDetectionToRoi.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
  Rect _palmDetectionToRoi(Detection detection, ImageSize imageSize) {
    // Converts results of palm detection into a rectangle (normalized by image
    // size) that encloses the palm and is rotated such that the line connecting
    // center of the wrist and MCP of the middle finger is aligned with the
    // Y-axis of the rectangle.
    // PalmDetectionDetectionToRoi: DetectionsToRectsCalculator.
    final rawRoi = calculateDetectionsToRects(
        detection, ConversionMode.boundingbox,
        normalized: true,
        imageSize: imageSize,
        rotationConfig: DetectionToRectConfig(
            rotationVectorStartKeypointIndex: 0,
            rotationVectorEndKeypointIndex: 2,
            rotationVectorTargetAngleDegree: 90));

    // Expands and shifts the rectangle that contains the palm so that it's
    // likely to cover the entire hand.
    // PalmDetectionDetectionToRoi: RectTransformationCalculation.
    final roi = transformNormalizedRect(rawRoi, imageSize,
        constants.MPHANDS_DETECTOR_RECT_TRANSFORMATION_CONFIG);

    return roi;
  }

  // Predict hand landmarks.
  // subgraph: HandLandmarkCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt
  Future<HandLandmarksResult?> _handLandmarks(
      Rect handRect, tf.Tensor3D image) async {
    // HandLandmarkCpu: ImageToTensorCalculator
    // Transforms a region of image into a 224x224 tensor while keeping the
    // aspect ratio, and therefore may result in potential letterboxing.
    final _imageTensor = convertImageToTensor(
        image, constants.MPHANDS_LANDMARK_IMAGE_TO_TENSOR_CONFIG, handRect);

    final imageValueShifted = _imageTensor.imageTensor;
    final padding = _imageTensor.padding;

    // HandLandmarkCpu: InferenceCalculator
    // Runs a model takes an image tensor and
    // outputs a list of tensors representing, for instance, detection
    // boxes/keypoints and scores.
    // The model returns 3 tensors with the following shape:
    // Identity_2:0: This tensor (shape: [1, 63]) represents 21 3-d
    // keypoints.
    // Identity_1:0: This tensor (shape: [1, 1]) represents the
    // confidence score of the presence of a hand.
    // Identity:0: This tensor (shape: [1, 1]) represents the classication
    // score of handedness
    // Identity:3: This tensor (shape: [1, 63]) represents 21 3DWorld keypoints.
    final landmarkResult = this.landmarkModel.execute(imageValueShifted, [
      'Identity_2:0',
      'Identity_1:0',
      'Identity:0',
      'Identity_3:0'
    ]) as List<tf.Tensor>;

    final landmarkTensor = landmarkResult[0] as tf.Tensor2D,
        handFlagTensor = landmarkResult[1] as tf.Tensor2D,
        handednessTensor = landmarkResult[2] as tf.Tensor2D,
        worldLandmarkTensor = landmarkResult[3] as tf.Tensor2D;

    // Converts the hand-flag tensor into a float that represents the
    // confidence score of pose presence.
    final handScore = (await handFlagTensor.data())[0];

    // Applies a threshold to the confidence score to determine whether a hand
    // is present.
    if (handScore < constants.MPHANDS_HAND_PRESENCE_SCORE) {
      tf.dispose(landmarkResult);
      tf.dispose(imageValueShifted);

      return null;
    }

    // Converts the handedness tensor into a float that represents the
    // classification score of handedness.
    final handednessScore = (await handednessTensor.data())[0];
    final handedness =
        handednessScore >= 0.5 ? Handedness.left : Handedness.right;

    // Decodes the landmark tensors into a list of landmarks, where the
    // landmark coordinates are normalized by the size of the input image to
    // the model.
    // HandLandmarkCpu: TensorsToLandmarksCalculator.
    final landmarks = await tensorsToLandmarks(
        landmarkTensor, constants.MPHANDS_TENSORS_TO_LANDMARKS_CONFIG);

    // Decodes the landmark tensors into a list of landmarks, where the landmark
    // coordinates are normalized by the size of the input image to the model.
    // HandLandmarkCpu: TensorsToLandmarksCalculator.
    final worldLandmarks = await tensorsToLandmarks(worldLandmarkTensor,
        constants.MPHANDS_TENSORS_TO_WORLD_LANDMARKS_CONFIG);

    // Adjusts landmarks (already normalized to [0.0, 1.0]) on the letterboxed
    // hand image to the corresponding locations on the same image with the
    // letterbox removed.
    // HandLandmarkCpu: LandmarkLetterboxRemovalCalculator.
    final adjustedLandmarks = removeLandmarkLetterbox(landmarks, padding);

    // Projects the landmarks from the cropped hand image to the corresponding
    // locations on the full image before cropping (input to the graph).
    // HandLandmarkCpu: LandmarkProjectionCalculator.
    final landmarksProjected =
        calculateLandmarkProjection(adjustedLandmarks, handRect);

    // Projects the world landmarks from the cropped pose image to the
    // corresponding locations on the full image before cropping (input to the
    // graph).
    // HandLandmarkCpu: WorldLandmarkProjectionCalculator.
    final worldLandmarksProjected =
        calculateWorldLandmarkProjection(worldLandmarks, handRect);

    tf.dispose(landmarkResult);
    tf.dispose(imageValueShifted);

    return HandLandmarksResult(
      landmarks: landmarksProjected,
      worldLandmarks: worldLandmarksProjected,
      handScore: handScore,
      handedness: handedness,
    );
  }

  // Calculate hand region of interest (ROI) from landmarks.
  // Subgraph: HandLandmarkLandmarksToRoi
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt
  // When landmarks is not null, imageSize should not be null either.
  Rect _handLandmarksToRoi(List<Keypoint> landmarks, ImageSize imageSize) {
    // Extracts a subset of the hand landmarks that are relatively more stable
    // across frames (e.g. comparing to finger tips) for computing the bounding
    // box. The box will later be expanded to contain the entire hand. In this
    // approach, it is more robust to drastically changing hand size. The
    // landmarks extracted are: wrist, MCP/PIP of five fingers.
    // HandLandmarkLandmarksToRoi: SplitNormalizedLandmarkListCalculator.
    final partialLandmarks = [
      landmarks.slice(0, 4),
      landmarks.slice(5, 7),
      landmarks.slice(9, 11),
      landmarks.slice(13, 15),
      landmarks.slice(17, 19)
    ].expand((l) => l).toList();

    // Converts the hand landmarks into a rectangle (normalized by image size)
    // that encloses the hand. The calculator uses a subset of all hand
    // landmarks extracted from the concat + slice above to
    // calculate the bounding box and the rotation of the output rectangle.
    // HandLandmarkLandmarksToRoi: HandLandmarksToRectCalculator.
    final rawRoi = handLandmarksToRect(partialLandmarks, imageSize);

    // Expands pose rect with marging used during training.
    // PoseLandmarksToRoi: RectTransformationCalculator.
    final roi = transformNormalizedRect(rawRoi, imageSize,
        constants.MPHANDS_LANDMARK_RECT_TRANSFORMATION_CONFIG);

    return roi;
  }
}

/**
 * Loads the MediaPipeHands model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeHands loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeHandsTfjsModelConfig`
 * interface.
 */
Future<HandDetector> load(MediaPipeHandsTfjsModelConfig modelConfig) async {
  final config = validateModelConfig(modelConfig);

  final detectorFromTFHub =
      config.detectorModelUrl!.contains('https://tfhub.dev');
  final landmarkFromTFHub =
      config.landmarkModelUrl!.contains('https://tfhub.dev');

  final models = await Future.wait([
    tfconv.loadGraphModel(tfconv.ModelHandler.fromUrl(config.detectorModelUrl!),
        tfconv.LoadOptions(fromTFHub: detectorFromTFHub)),
    tfconv.loadGraphModel(tfconv.ModelHandler.fromUrl(config.landmarkModelUrl!),
        tfconv.LoadOptions(fromTFHub: landmarkFromTFHub))
  ]);

  return MediaPipeHandsTfjsDetector(models[0], models[1], config.maxHands!);
}
