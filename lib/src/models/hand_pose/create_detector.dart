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

import 'hand_detector.dart';
import 'tfjs/types.dart';
import 'types.dart';
import 'tfjs/detector.dart' as tfjs_detector;

// import {HandDetector} from './hand_detector';
// import {load as loadMediaPipeHandsMediaPipeDetector} from './mediapipe/detector';
// import {MediaPipeHandsMediaPipeModelConfig, MediaPipeHandsModelConfig} from './mediapipe/types';
// import {load as loadMediaPipeHandsTfjsDetector} from './tfjs/detector';
// import {MediaPipeHandsTfjsModelConfig} from './tfjs/types';
// import {SupportedModels} from './types';

/**
 * Create a hand detector instance.
 *
 * @param model The name of the pipeline to load.
 * @param modelConfig The configuration for the pipeline to load.
 */
Future<HandDetector> createDetector(
  SupportedModels model,
  MediaPipeHandsTfjsModelConfig?
      modelConfig, // MediaPipeHandsMediaPipeModelConfig| MediaPipeHandsTfjsModelConfig,
) async {
  switch (model) {
    case SupportedModels.mediaPipeHands:
      final config = modelConfig;
      if (config != null) {
        if (config.runtime == 'tfjs') {
          return tfjs_detector.load(config);
        }
        if (config.runtime == 'mediapipe') {
          // return loadMediaPipeHandsMediaPipeDetector(
          //     config as MediaPipeHandsMediaPipeModelConfig);
          throw UnimplementedError(
              "Expect modelConfig.runtime 'mediapipe' not implemented.");
        }
      }
      throw Exception("Expect modelConfig.runtime to be either 'tfjs' " +
          "or 'mediapipe', but got ${config?.runtime}");
    default:
      throw Exception("${model} is not a supported model name.");
  }
}
