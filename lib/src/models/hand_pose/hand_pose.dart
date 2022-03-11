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

export './create_detector.dart' show createDetector;
// HandDetector class.
export './hand_detector.dart' show HandDetector;
// Entry point to create a new detector instance.
// export {MediaPipeHandsMediaPipeEstimationConfig, MediaPipeHandsMediaPipeModelConfig, MediaPipeHandsModelType} from './mediapipe/types.dart';
export './tfjs/types.dart'
    show
        MediaPipeHandsTfjsEstimationConfig,
        MediaPipeHandsTfjsModelConfig,
        MediaPipeHandsModelType;

// Supported models enum.
export './types.dart';
