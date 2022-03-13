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
// import {FaceLandmarksDetector} from '../face_landmarks_detector';
// import {calculateAssociationNormRect} from '../shared/calculators/association_norm_rect';
// import {calculateLandmarkProjection} from '../shared/calculators/calculate_landmark_projection';
// import {convertImageToTensor} from '../shared/calculators/convert_image_to_tensor';
// import {createSsdAnchors} from '../shared/calculators/create_ssd_anchors';
// import {detectionProjection} from '../shared/calculators/detection_projection';
// import {calculateDetectionsToRects} from '../shared/calculators/detection_to_rect';
// import {detectorResult} from '../shared/calculators/detector_result';
// import {getImageSize, toImageTensor} from '../shared/calculators/image_utils';
// import {ImageSize, Keypoint} from '../shared/calculators/interfaces/common_interfaces';
// import {ImageToTensorConfig, TensorsToDetectionsConfig} from '../shared/calculators/interfaces/config_interfaces';
// import {Rect} from '../shared/calculators/interfaces/shape_interfaces';
// import {AnchorTensor, Detection} from '../shared/calculators/interfaces/shape_interfaces';
// import {landmarksRefinement} from '../shared/calculators/landmarks_refinement';
// import {landmarksToDetection} from '../shared/calculators/landmarks_to_detection';
// import {nonMaxSuppression} from '../shared/calculators/non_max_suppression';
// import {normalizedKeypointsToKeypoints} from '../shared/calculators/normalized_keypoints_to_keypoints';
// import {tensorsToDetections} from '../shared/calculators/tensors_to_detections';
// import {tensorsToLandmarks} from '../shared/calculators/tensors_to_landmarks';
// import {transformNormalizedRect} from '../shared/calculators/transform_rect';
// import {Face, FaceLandmarksDetectorInput} from '../types';

// import * as constants from './constants';
// import {validateDetectorModelConfig, validateEstimationConfig, validateMeshModelConfig} from './detector_utils';
// import {MediaPipeFaceDetectorTfjsModelConfig, MediaPipeFaceMeshTfjsEstimationConfig, MediaPipeFaceMeshTfjsModelConfig} from './types';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;
import '../../shared/association_norm_rect.dart';
import '../../shared/calculate_landmark_projection.dart';
import '../../shared/convert_image_to_tensor.dart';
import '../../shared/create_ssd_anchors.dart';
import '../../shared/detection_projection.dart';
import '../../shared/detection_to_rect.dart';
import '../../shared/detector_result.dart';
import '../../shared/image_utils.dart';
import '../../shared/interfaces/common_interfaces.dart';
import '../../shared/interfaces/config_interfaces.dart';
import '../../shared/interfaces/shape_interfaces.dart';
import '../../shared/landmarks_refinement.dart';
import '../../shared/landmarks_to_detection.dart';
import '../../shared/non_max_suppression.dart';
import '../../shared/normalized_keypoints_to_keypoints.dart';
import '../../shared/tensors_to_detections.dart';
import '../../shared/tensors_to_landmarks.dart';
import '../../shared/transform_rect.dart';
import '../constants.dart';
import '../face_landmarks_detector.dart';
import '../types.dart';
import 'constants.dart' as constants;
import 'types.dart';

class MediaPipeFaceDetectorTfjs {
  // TODO: all private
  late final ImageToTensorConfig imageToTensorConfig;
  late final TensorsToDetectionsConfig tensorsToDetectionConfig;
  late final List<Rect> anchors;
  late final AnchorTensor anchorTensor;

  final MediaPipeFaceDetectorModelType detectorModelType;
  final tfconv.GraphModel detectorModel;
  final int maxFaces;

  MediaPipeFaceDetectorTfjs(
    this.detectorModelType,
    this.detectorModel,
    this.maxFaces,
  ) {
    if (detectorModelType == MediaPipeFaceDetectorModelType.full) {
      this.imageToTensorConfig = constants.FULL_RANGE_IMAGE_TO_TENSOR_CONFIG;
      this.tensorsToDetectionConfig =
          constants.FULL_RANGE_TENSORS_TO_DETECTION_CONFIG;
      this.anchors =
          createSsdAnchors(constants.FULL_RANGE_DETECTOR_ANCHOR_CONFIG);
    } else {
      this.imageToTensorConfig = constants.SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG;
      this.tensorsToDetectionConfig =
          constants.SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG;
      this.anchors =
          createSsdAnchors(constants.SHORT_RANGE_DETECTOR_ANCHOR_CONFIG);
    }

    final anchorW = tf.tensor1d(this.anchors.map((a) => a.width).toList());
    final anchorH = tf.tensor1d(this.anchors.map((a) => a.height).toList());
    final anchorX = tf.tensor1d(this.anchors.map((a) => a.xCenter).toList());
    final anchorY = tf.tensor1d(this.anchors.map((a) => a.yCenter).toList());
    this.anchorTensor =
        AnchorTensor(x: anchorX, y: anchorY, w: anchorW, h: anchorH);
  }

  void dispose() {
    this.detectorModel.dispose();
    tf.dispose([
      this.anchorTensor.x,
      this.anchorTensor.y,
      this.anchorTensor.w,
      this.anchorTensor.h
    ]);
  }

  void reset() {}

  // Detects faces.
  // Subgraph: FaceDetectionShort/FullRangeCpu.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_full_range_cpu.pbtxt
  Future<List<Detection>> detectFaces(FaceLandmarksDetectorInput image,
      {bool flipHorizontal = false}) async {
    if (image == null) {
      this.reset();
      return [];
    }

    final image3d = tf.tidy(() {
      var imageTensor = tf.cast(toImageTensor(image), 'float32');
      if (flipHorizontal) {
        final batchAxis = 0;
        imageTensor = tf.squeeze(tf.image.flipLeftRight(
            // tslint:disable-next-line: no-unnecessary-type-assertion
            tf.expandDims(imageTensor, batchAxis) as tf.Tensor4D), [batchAxis]);
      }
      return imageTensor;
    });

    // FaceDetectionShort/FullRangeModelCpu: ImageToTensorCalculator
    // Transforms the input image into a 128x128 tensor while keeping the aspect
    // ratio (what is expected by the corresponding face detection model),
    // resulting in potential letterboxing in the transformed image.
    final _img = convertImageToTensor(image3d, this.imageToTensorConfig, null);
    final transformationMatrix = _img.transformationMatrix;
    final inputTensors = _img.imageTensor;

    final detectionResult =
        this.detectorModel.execute(inputTensors, ['Identity:0']) as tf.Tensor3D;
    // FaceDetectionShort/FullRangeModelCpu: InferenceCalculator
    // The model returns a tensor with the following shape:
    // [1 (batch), 896 (anchor points), 17 (data for each anchor)]
    final results = detectorResult(detectionResult);
    // FaceDetectionShort/FullRangeModelCpu: TensorsToDetectionsCalculator
    final List<Detection> unfilteredDetections = await tensorsToDetections(
        [results.logits, results.boxes],
        this.anchorTensor,
        this.tensorsToDetectionConfig);

    if (unfilteredDetections.length == 0) {
      tf.dispose([
        image3d,
        inputTensors,
        detectionResult,
        results.logits,
        results.boxes
      ]);
      return unfilteredDetections;
    }

    // FaceDetectionShort/FullRangeModelCpu: NonMaxSuppressionCalculator
    final filteredDetections = await nonMaxSuppression(
      unfilteredDetections,
      this.maxFaces,
      constants.DETECTOR_NON_MAX_SUPPRESSION_CONFIG.minSuppressionThreshold,
      // constants.DETECTOR_NON_MAX_SUPPRESSION_CONFIG.overlapType,
    );

    final detections =
        // FaceDetectionShortRangeModelCpu:
        // DetectionProjectionCalculator
        detectionProjection(filteredDetections, transformationMatrix);

    tf.dispose([
      image3d,
      inputTensors,
      detectionResult,
      results.logits,
      results.boxes
    ]);

    return detections;
  }
}

/**
 * Loads the MediaPipeFaceDetector model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeFaceDetector loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeHandsTfjsModelConfig`
 * interface.
 */
Future<MediaPipeFaceDetectorTfjs> loadDetectorModel(
    MediaPipeFaceDetectorTfjsModelConfig config) async {
  // final config = validateDetectorModelConfig(modelConfig);

  final detectorFromTFHub = config.detectorModelUrl.isUrl &&
      config.detectorModelUrl.url!.contains('https://tfhub.dev');

  final detectorModel = await tfconv.loadGraphModel(
    config.detectorModelUrl,
    tfconv.LoadOptions(fromTFHub: detectorFromTFHub),
  );

  return MediaPipeFaceDetectorTfjs(
      config.modelType, detectorModel, config.maxFaces);
}

/**
 * MediaPipFaceMesh class.
 */
class MediaPipeFaceMeshTfjsLandmarksDetector implements FaceLandmarksDetector {
  // TODO: all private
  // Store global states.
  List<Rect>? prevFaceRectsFromLandmarks;

  final MediaPipeFaceDetectorTfjs detector;
  final tfconv.GraphModel landmarkModel;
  final int maxFaces;
  final bool withAttention;

  MediaPipeFaceMeshTfjsLandmarksDetector(
    this.detector,
    this.landmarkModel,
    this.maxFaces,
    this.withAttention,
  );

  /**
   * Estimates faces for an image or video frame.
   *
   * It returns a single face or multiple faces based on the maxFaces
   * parameter from the `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param estimationConfig Optional. See
   *     `MediaPipeFaceMeshTfjsEstimationConfig` documentation for detail.
   *
   * @return An array of `Face`s.
   */
  // TF.js implementation of the mediapipe face landmark pipeline.
  // ref graph:
  // https://github.com/google/mediapipe/blob/master/mediapipe/mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt
  Future<List<Face>> estimateFaces(
    FaceLandmarksDetectorInput image, [
    MediaPipeFaceMeshTfjsEstimationConfig? config_,
  ]) async {
    final config = config_ ?? const MediaPipeFaceMeshTfjsEstimationConfig();
    // final config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      this.reset();
      return [];
    }

    // FaceLandmarkFrontCpu: ImagePropertiesCalculator
    // Calculate size of the image.
    final imageSize = getImageSize(image);

    final image3d = tf.tidy(() {
      var imageTensor = tf.cast(toImageTensor(image), 'float32');
      if (config.flipHorizontal) {
        final batchAxis = 0;
        imageTensor = tf.squeeze(tf.image.flipLeftRight(
            // tslint:disable-next-line: no-unnecessary-type-assertion
            tf.expandDims(imageTensor, batchAxis) as tf.Tensor4D), [batchAxis]);
      }
      return imageTensor;
    });

    final prevFaceRectsFromLandmarks = this.prevFaceRectsFromLandmarks;

    final List<Rect> faceRectsFromDetections;
    // Drops the incoming image if enough faces have already been identified
    // from the previous image. Otherwise, passes the incoming image through to
    // trigger a new round of face detection.
    if (config.staticImageMode ||
        prevFaceRectsFromLandmarks == null ||
        prevFaceRectsFromLandmarks.length < this.maxFaces) {
      // FaceLandmarkFrontCpu: FaceDetectionShortRangeCpu
      // Detects faces.
      final allFaceDetections = await this.detector.detectFaces(image3d);

      if (allFaceDetections.length == 0) {
        this.reset();
        image3d.dispose();
        return [];
      }

      // FaceLandmarkFrontCpu: ClipDetectionVectorSizeCalculator
      // Makes sure there are no more detections than the provided maxFaces.
      // This is already done by our implementation of nonMaxSuppresion.
      final faceDetections = allFaceDetections;

      // FaceLandmarkFrontCpu: FaceDetectionFrontDetectionToRoi
      // Calculates region of interest based on face detections, so that can be
      // used to detect landmarks.
      faceRectsFromDetections = faceDetections
          .map((detection) =>
              this._faceDetectionFrontDetectionToRoi(detection, imageSize))
          .toList();
    } else {
      faceRectsFromDetections = [];
    }

    // FaceLandmarkFrontCpu: AssociationNormRectCalculator
    // Performs association between NormalizedRect vector elements from
    // previous image and rects based on face detections from the current image.
    // This calculator ensures that the output faceRects array doesn't contain
    // overlapping regions based on the specified minSimilarityThreshold.
    final faceRects = calculateAssociationNormRect(
        [faceRectsFromDetections, prevFaceRectsFromLandmarks ?? []],
        constants.MIN_SIMILARITY_THRESHOLD);

    // FaceLandmarkFrontCpu: FaceLandmarkCpu
    // Detects face landmarks within specified region of interest of the image.
    final faceLandmarks = await Future.wait(
        faceRects.map((faceRect) => this._faceLandmark(faceRect, image3d)));

    final faces = <Face>[];
    this.prevFaceRectsFromLandmarks = [];

    for (int i = 0; i < faceLandmarks.length; ++i) {
      final landmarks = faceLandmarks[i];

      if (landmarks == null) {
        continue;
      }

      this
          .prevFaceRectsFromLandmarks!
          .add(this._faceLandmarksToRoi(landmarks, imageSize));

      // Scale back keypoints.
      var keypoints = normalizedKeypointsToKeypoints(landmarks, imageSize);

      // Add keypoint name.
      if (keypoints != null) {
        keypoints = keypoints.mapIndexed((i, keypoint) {
          final name = MEDIAPIPE_KEYPOINTS.get(i);
          return keypoint.copyWith(name: Nullable(name ?? keypoint.name));
        }).toList();
      }

      final detection = landmarksToDetection(keypoints);

      faces.add(Face(
          keypoints: keypoints,
          box: detection.locationData.relativeBoundingBox));
    }

    image3d.dispose();

    return faces;
  }

  void dispose() {
    this.detector.dispose();
    this.landmarkModel.dispose();
  }

  void reset() {
    this.detector.reset();
    this.prevFaceRectsFromLandmarks = null;
  }

  // calculates face ROI from face detection.
  // Subgraph: FaceDetectionFrontDetectionToRoi.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
  Rect _faceDetectionFrontDetectionToRoi(
      Detection detection, ImageSize imageSize) {
    // Converts results of face detection into a rectangle (normalized by
    // image size) that encloses the face and is rotated such that the line
    // connecting left eye and right eye is aligned with the X-axis of the
    // rectangle.
    // FaceDetectionFrontDetectionToRoi: DetectionsToRectsCalculator.
    final rawRoi =
        calculateDetectionsToRects(detection, ConversionMode.boundingbox,
            normalized: true,
            imageSize: imageSize,
            rotationConfig: DetectionToRectConfig(
                rotationVectorStartKeypointIndex: 0, // Left eye.
                rotationVectorEndKeypointIndex: 1, // Right eye.
                rotationVectorTargetAngleDegree: 0));

    // Expands and shifts the rectangle that contains the face so that it's
    // likely to cover the entire face.
    // FaceDetectionFrontDetectionToRoi: RectTransformationCalculation.
    final roi = transformNormalizedRect(
        rawRoi, imageSize, constants.RECT_TRANSFORMATION_CONFIG);

    return roi;
  }

  // Predict face landmarks.
  // subgraph: FaceLandmarkCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt
  Future _faceLandmark(Rect roi, tf.Tensor3D image) async {
    // FaceLandmarkCpu: ImageToTensorCalculator
    // Transforms the input image into a 192x192 tensor.
    final _img = convertImageToTensor(
        image, constants.LANDMARK_IMAGE_TO_TENSOR_CONFIG, roi);
    final inputTensors = _img.imageTensor;

    // FaceLandmarkCpu: InferenceCalculator
    // Runs a model takes an image tensor and
    // outputs a list of tensors representing, for instance, detection
    // boxes/keypoints and scores.
    final outputs = [
      'output_faceflag',
      ...(this.withAttention
          ? [
              'output_mesh_identity',
              'output_lips',
              'Identity_6:0',
              'Identity_1:0',
              'Identity_2:0',
              'Identity_5:0'
            ]
          : ['output_mesh']),
    ];
    // The model returns 2 or 7 tensors with the following shape:
    // output_faceflag: This tensor (shape: [1, 1]) represents the
    // confidence score of the presence of a face.
    // Other outputs represents 2-d or 3-d keypoints of different parts of the
    // face.
    final outputTensors =
        this.landmarkModel.execute(inputTensors, outputs) as List<tf.Tensor>;

    final faceFlagTensor = outputTensors[0] as tf.Tensor2D,
        landmarkTensors = outputTensors.slice(1) as List<tf.Tensor4D>;

    // Converts the face-flag tensor into a float that represents the
    // confidence score of face presence.
    final facePresenceScore = (await faceFlagTensor.data())[0];

    // Applies a threshold to the confidence score to determine whether a face
    // is present.
    if (facePresenceScore < constants.FACE_PRESENCE_SCORE) {
      tf.dispose(outputTensors);
      tf.dispose(inputTensors);

      return null;
    }

    // Decodes the landmark tensors into a list of landmarks, where the
    // landmark coordinates are normalized by the size of the input image to
    // the model.
    // FaceLandmarkCpu: TensorsToFaceLandmarks /
    // TensorsToFaceLandmarksWithAttention.
    final landmarks = this.withAttention
        ? await this.tensorsToFaceLandmarksWithAttention(landmarkTensors)
        : await this._tensorsToFaceLandmarks(landmarkTensors);

    // Projects the landmarks from the cropped face image to the corresponding
    // locations on the full image before cropping.
    // FaceLandmarkCpu: WorldLandmarkProjectionCalculator.
    final faceLandmarks = calculateLandmarkProjection(landmarks, roi);

    tf.dispose(outputTensors);
    tf.dispose(inputTensors);

    return faceLandmarks;
  }

  // Transform single tensor into 468 facial landmarks.
  // subgraph: TensorsToFaceLandmarks
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt
  Future<List<Keypoint>> _tensorsToFaceLandmarks(
      List<tf.Tensor4D> landmarkTensors) async {
    return tensorsToLandmarks(
        landmarkTensors[0], constants.TENSORS_TO_LANDMARKS_MESH_CONFIG);
  }

  // Transform model output tensors into 478 facial landmarks with refined
  // lips, eyes and irises.
  // subgraph: TensorsToFaceLandmarks
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt
  Future tensorsToFaceLandmarksWithAttention(
      List<tf.Tensor4D> landmarkTensors) async {
    final meshLandmarks = await tensorsToLandmarks(
        landmarkTensors[0], constants.TENSORS_TO_LANDMARKS_MESH_CONFIG);
    final lipsLandmarks = await tensorsToLandmarks(
        landmarkTensors[1], constants.TENSORS_TO_LANDMARKS_LIPS_CONFIG);
    final leftEyeLandmarks = await tensorsToLandmarks(
        landmarkTensors[3], constants.TENSORS_TO_LANDMARKS_EYE_CONFIG);
    final rightEyeLandmarks = await tensorsToLandmarks(
        landmarkTensors[5], constants.TENSORS_TO_LANDMARKS_EYE_CONFIG);
    final leftIrisLandmarks = await tensorsToLandmarks(
        landmarkTensors[4], constants.TENSORS_TO_LANDMARKS_IRIS_CONFIG);
    final rightIrisLandmarks = await tensorsToLandmarks(
        landmarkTensors[2], constants.TENSORS_TO_LANDMARKS_IRIS_CONFIG);

    return landmarksRefinement([
      meshLandmarks,
      lipsLandmarks,
      leftEyeLandmarks,
      rightEyeLandmarks,
      leftIrisLandmarks,
      rightIrisLandmarks
    ], [
      constants.LANDMARKS_REFINEMENT_MESH_CONFIG,
      constants.LANDMARKS_REFINEMENT_LIPS_CONFIG,
      constants.LANDMARKS_REFINEMENT_LEFT_EYE_CONFIG,
      constants.LANDMARKS_REFINEMENT_RIGHT_EYE_CONFIG,
      constants.LANDMARKS_REFINEMENT_LEFT_IRIS_CONFIG,
      constants.LANDMARKS_REFINEMENT_RIGHT_IRIS_CONFIG
    ]);
  }

  // Calculate face region of interest (ROI) from detections.
  // subgraph: FaceLandmarkLandmarksToRoi
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt
  Rect _faceLandmarksToRoi(List<Keypoint> landmarks, ImageSize imageSize) {
    // Converts face landmarks to a detection that tightly encloses all
    // landmarks.
    // FaceLandmarkLandmarksToRoi: LandmarksToDetectionCalculator.
    final faceDetection = landmarksToDetection(landmarks);
    // Converts the face detection into a rectangle (normalized by image size)
    // that encloses the face and is rotated such that the line connecting
    // left side of the left eye and right side of the right eye is aligned
    // with the X-axis of the rectangle.
    // FaceLandmarkLandmarksToRoi: DetectionsToRectsCalculator
    final faceRectFromLandmarks =
        calculateDetectionsToRects(faceDetection, ConversionMode.boundingbox,
            normalized: true,
            imageSize: imageSize,
            rotationConfig: DetectionToRectConfig(
              rotationVectorStartKeypointIndex: 33, // Left side of left eye.
              rotationVectorEndKeypointIndex: 263, // Right side of right eye.
              rotationVectorTargetAngleDegree: 0,
            ));

    // Expands the face rectangle so that in the next video image it's likely
    // to still contain the face even with some motion.
    // FaceLandmarkLandmarksToRoi: RectTransformationCalculator.
    // TODO: `squareLong` in the config should be set to false in MediaPipe code
    // but is not due to a bug in their processing. Once fixed on their end,
    // split RECT_TRANSFORMATION_CONFIG into separate detector and landmark
    // configs, with landmark's config's `squareLong` set to false.
    final roi = transformNormalizedRect(
        faceRectFromLandmarks, imageSize, constants.RECT_TRANSFORMATION_CONFIG);

    return roi;
  }
}

/**
 * Loads the MediaPipeFaceMesh model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeFaceMesh loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeHandsTfjsModelConfig`
 * interface.
 */
Future<FaceLandmarksDetector> loadMeshModel(
    MediaPipeFaceMeshTfjsModelConfig config) async {
  // final config = validateMeshModelConfig(modelConfig);

  final landmarkFromTFHub = config.landmarkModelUrl.isUrl &&
      config.landmarkModelUrl.url!.contains('https://tfhub.dev');
  final landmarkModel = await tfconv.loadGraphModel(config.landmarkModelUrl,
      tfconv.LoadOptions(fromTFHub: landmarkFromTFHub));

  final detector = await loadDetectorModel(MediaPipeFaceDetectorTfjsModelConfig(
      modelType: MediaPipeFaceDetectorModelType.short,
      maxFaces: config.maxFaces,
      detectorModelUrl: config.detectorModelUrl));

  return MediaPipeFaceMeshTfjsLandmarksDetector(
      detector, landmarkModel, config.maxFaces, config.refineLandmarks);
}
