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

// import {FaceLandmarksDetector} from './face_landmarks_detector';
// import {load as loadMediaPipeFaceMeshMediaPipeLandmarksDetector} from './mediapipe/detector';
// import {MediaPipeFaceMeshMediaPipeModelConfig, MediaPipeFaceMeshModelConfig} from './mediapipe/types';
// import {loadMeshModel as loadMediaPipeFaceMeshTfjsLandmarksDetector} from './tfjs/detector';
// import {MediaPipeFaceMeshTfjsModelConfig} from './tfjs/types';
// import {SupportedModels} from './types';

import 'face_landmarks_detector.dart';
import 'tfjs/detector.dart';
import 'tfjs/types.dart';
import 'types.dart';

/**
 * Create a face detector instance.
 *
 * @param model The name of the pipeline to load.
 * @param modelConfig The configuration for the pipeline to load.
 */
Future<FaceLandmarksDetector> createDetector(
  SupportedModels model, [
  MediaPipeFaceMeshTfjsModelConfig? modelConfig,
] // : MediaPipeFaceMeshMediaPipeModelConfig| MediaPipeFaceMeshTfjsModelConfig
    ) {
  switch (model) {
    case SupportedModels.MediaPipeFaceMesh:
      final config = modelConfig;
      String? runtime;
      if (config != null) {
        if (config.runtime == 'tfjs') {
          return loadMeshModel(config);
        }
        // if (config.runtime == 'mediapipe') {
        //   return loadMediaPipeFaceMeshMediaPipeLandmarksDetector(
        //       config as MediaPipeFaceMeshMediaPipeModelConfig);
        // }
        runtime = config.runtime;
      }
      throw Exception("Expect modelConfig.runtime to be either 'tfjs' " +
          "or 'mediapipe', but got ${runtime}");
    default:
      throw Exception("${model} is not a supported model name.");
  }
}
