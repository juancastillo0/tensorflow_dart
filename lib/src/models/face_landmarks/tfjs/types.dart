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

// import {io} from '@tensorflow/tfjs-core';

// import {MediaPipeFaceMeshEstimationConfig, MediaPipeFaceMeshModelConfig} from '../mediapipe/types';

import 'package:tensorflow_wasm/converter.dart';

import '../types.dart';
import 'constants.dart';

enum MediaPipeFaceDetectorModelType {
  short,
  full,
}

/**
 * Model parameters for MediaPipeFaceDetector TFJS runtime.
 *
 * `modelType`: Optional. Possible values: 'short'|'full'. Defaults to
 * 'short'. The short-range model that works best for faces within 2 meters from
 * the camera, while the full-range model works best for faces within 5 meters.
 * For the full-range option, a sparse model is used for its improved inference
 * speed.
 *
 * `maxFaces`: Optional. Default to 1. The maximum number of faces that will
 * be detected by the model. The number of returned faces can be less than the
 * maximum (for example when no faces are present in the input).
 *
 * `detectorModelUrl`: Optional. An optional string that specifies custom url of
 * the detector model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 */
class MediaPipeFaceDetectorTfjsModelConfig {
  final MediaPipeFaceDetectorModelType modelType;
  final int maxFaces;
  final ModelHandler detectorModelUrl;

  const MediaPipeFaceDetectorTfjsModelConfig({
    this.modelType = MediaPipeFaceDetectorModelType.short,
    this.maxFaces = 1,
    ModelHandler? detectorModelUrl,
  }) : detectorModelUrl = detectorModelUrl ??
            (modelType == MediaPipeFaceDetectorModelType.full
                ? const ModelHandler.fromUrl(
                    DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE)
                : const ModelHandler.fromUrl(DEFAULT_DETECTOR_MODEL_URL_SHORT));
}

/**
 * Model parameters for MediaPipeFaceMesh TFJS runtime.
 *
 * `runtime`: Must set to be 'tfjs'.
 *
 * `refineLandmarks`: Defaults to false. If set to true, refines the landmark
 * coordinates around the eyes and lips, and output additional landmarks around
 * the irises.
 *
 * `detectorModelUrl`: Optional. An optional string that specifies custom url of
 * the detector model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 *
 * `landmarkModelUrl`: Optional. An optional string that specifies custom url of
 * the landmark model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 */
class MediaPipeFaceMeshTfjsModelConfig implements MediaPipeFaceMeshModelConfig {
  @override
  String get runtime => 'tfjs';
  @override
  final int maxFaces;
  @override
  final bool refineLandmarks;

  final ModelHandler detectorModelUrl;
  final ModelHandler landmarkModelUrl;

  const MediaPipeFaceMeshTfjsModelConfig({
    this.detectorModelUrl =
        const ModelHandler.fromUrl(DEFAULT_DETECTOR_MODEL_URL_SHORT),
    ModelHandler? landmarkModelUrl,
    this.maxFaces = 1,
    this.refineLandmarks = false,
  }) : landmarkModelUrl = landmarkModelUrl ??
            (refineLandmarks
                ? const ModelHandler.fromUrl(
                    DEFAULT_LANDMARK_MODEL_URL_WITH_ATTENTION)
                : const ModelHandler.fromUrl(DEFAULT_LANDMARK_MODEL_URL));
}

/**
 * Face estimation parameters for MediaPipeFaceMesh TFJS runtime.
 */
class MediaPipeFaceMeshTfjsEstimationConfig extends EstimationConfig {
  final bool flipHorizontal;
  final bool staticImageMode;

  const MediaPipeFaceMeshTfjsEstimationConfig({
    this.flipHorizontal = false,
    this.staticImageMode = false,
  });
}

/**
 * Common MediaPipeFaceMesh model config.
 */
abstract class MediaPipeFaceMeshModelConfig extends ModelConfig {
  String get runtime;
  bool get refineLandmarks;
}
