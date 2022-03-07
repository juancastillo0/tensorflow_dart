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

// import {MediaPipeHandsEstimationConfig, MediaPipeHandsModelConfig} from '../mediapipe/types';

import '../types.dart';

/**
 * Model parameters for MediaPipeHands TFJS runtime.
 *
 * `runtime`: Must set to be 'tfjs'.
 *
 * `modelType`: Optional. Possible values: 'lite'|'full'. Defaults to
 * 'full'. Landmark accuracy as well as inference latency generally go up with
 * the increasing model complexity (lite to full).
 *
 * `detectorModelUrl`: Optional. An optional string that specifies custom url of
 * the detector model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 *
 * `landmarkModelUrl`: Optional. An optional string that specifies custom url of
 * the landmark model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 */
class MediaPipeHandsTfjsModelConfig implements ModelConfig {
  String get runtime => 'tfjs';

  final int? maxHands;
  final String? detectorModelUrl;
  final String? landmarkModelUrl;
  final MediaPipeHandsModelType? modelType;

  const MediaPipeHandsTfjsModelConfig({
    this.maxHands,
    this.detectorModelUrl,
    this.landmarkModelUrl,
    this.modelType,
  });

  MediaPipeHandsTfjsModelConfig copyWith({
    int? maxHands,
    String? detectorModelUrl,
    String? landmarkModelUrl,
    MediaPipeHandsModelType? modelType,
  }) {
    return MediaPipeHandsTfjsModelConfig(
      maxHands: maxHands ?? this.maxHands,
      detectorModelUrl: detectorModelUrl ?? this.detectorModelUrl,
      landmarkModelUrl: landmarkModelUrl ?? this.landmarkModelUrl,
      modelType: modelType ?? this.modelType,
    );
  }
}

enum MediaPipeHandsModelType {
  lite,
  full,
}

/**
 * Hand estimation parameters for MediaPipeHands TFJS runtime.
 */
typedef MediaPipeHandsTfjsEstimationConfig = EstimationConfig;
