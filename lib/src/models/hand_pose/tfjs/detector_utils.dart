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

// import {DEFAULT_MPHANDS_DETECTOR_MODEL_URL_FULL, DEFAULT_MPHANDS_DETECTOR_MODEL_URL_LITE, DEFAULT_MPHANDS_ESTIMATION_CONFIG, DEFAULT_MPHANDS_LANDMARK_MODEL_URL_FULL, DEFAULT_MPHANDS_LANDMARK_MODEL_URL_LITE, DEFAULT_MPHANDS_MODEL_CONFIG} from './constants';
// import {MediaPipeHandsTfjsEstimationConfig, MediaPipeHandsTfjsModelConfig} from './types';

import 'constants.dart';
import 'types.dart';

MediaPipeHandsTfjsModelConfig validateModelConfig(
    MediaPipeHandsTfjsModelConfig modelConfig) {
  // if (modelConfig == null) {
  //   return {...DEFAULT_MPHANDS_MODEL_CONFIG};
  // }

  final modelType =
      modelConfig.modelType ?? DEFAULT_MPHANDS_MODEL_CONFIG.modelType!;
  String? detectorModelUrl = modelConfig.detectorModelUrl;
  if (modelConfig.detectorModelUrl == null) {
    switch (modelType) {
      case MediaPipeHandsModelType.lite:
        detectorModelUrl = DEFAULT_MPHANDS_DETECTOR_MODEL_URL_LITE;
        break;
      case MediaPipeHandsModelType.full:
        detectorModelUrl = DEFAULT_MPHANDS_DETECTOR_MODEL_URL_FULL;
        break;
    }
  }
  String? landmarkModelUrl = modelConfig.landmarkModelUrl;
  if (modelConfig.landmarkModelUrl == null) {
    switch (modelType) {
      case MediaPipeHandsModelType.lite:
        landmarkModelUrl = DEFAULT_MPHANDS_LANDMARK_MODEL_URL_LITE;
        break;
      case MediaPipeHandsModelType.full:
      default:
        landmarkModelUrl = DEFAULT_MPHANDS_LANDMARK_MODEL_URL_FULL;
        break;
    }
  }
  final config = modelConfig.copyWith(
    maxHands: modelConfig.maxHands ?? DEFAULT_MPHANDS_MODEL_CONFIG.maxHands,
    modelType: modelType,
    detectorModelUrl: detectorModelUrl,
    landmarkModelUrl: landmarkModelUrl,
  );

  // if (config.modelType != 'lite' && config.modelType != 'full') {
  //   throw Exception(
  //       'Model type must be one of lite or full, but got ${config.modelType}');
  // }

  return config;
}

MediaPipeHandsTfjsEstimationConfig validateEstimationConfig(
    MediaPipeHandsTfjsEstimationConfig? estimationConfig) {
  // if (estimationConfig == null) {
  //   return {...DEFAULT_MPHANDS_ESTIMATION_CONFIG};
  // }

  final config = MediaPipeHandsTfjsEstimationConfig(
    flipHorizontal: estimationConfig?.flipHorizontal ??
        DEFAULT_MPHANDS_ESTIMATION_CONFIG.flipHorizontal,
    staticImageMode: estimationConfig?.staticImageMode ??
        DEFAULT_MPHANDS_ESTIMATION_CONFIG.staticImageMode,
  );

  return config;
}
