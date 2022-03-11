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
// import * as handdetection from '@tensorflow-models/hand-pose-detection';
import 'package:mobx/mobx.dart';
import 'package:tensorflow_wasm/models/hand_pose.dart' as handdetection;

const CANVAS_ELEMENT_ID = 'canvas-output';
const DEFAULT_LINE_WIDTH = 2;
const DEFAULT_RADIUS = 4;

const VIDEO_SIZE = {
  '640 X 480': {'width': 640, 'height': 480},
  '640 X 360': {'width': 640, 'height': 360},
  '360 X 270': {'width': 360, 'height': 270}
};
final STATE = HandPoseState(
  camera: CameraParams(targetFPS: 60, sizeOption: '640 X 480'),
  backend: '',
  flags: {},
  modelConfig: ModelConfig(),
);

class ModelConfig {
  handdetection.MediaPipeHandsModelType? type;
  int? maxNumHands;

  ModelConfig({
    this.type,
    this.maxNumHands,
  });
}

class HandPoseState {
  final CameraParams camera;
  final String backend;
  final Map<String, Object?> flags;
  final ModelConfig modelConfig;
  handdetection.SupportedModels get model =>
      handdetection.SupportedModels.mediaPipeHands;

  bool isTargetFPSChanged = false;
  bool isSizeOptionChanged = false;
  bool isModelChanged = false;
  bool isFlagChanged = false;
  bool isBackendChanged = false;

  HandPoseState({
    required this.camera,
    required this.backend,
    required this.flags,
    required this.modelConfig,
  });
}

class CameraParams {
  final Observable<String> sizeOption;
  final Observable<int> targetFPS;

  CameraParams({
    required String sizeOption,
    required int targetFPS,
  })  : sizeOption = Observable(sizeOption),
        targetFPS = Observable(targetFPS);
}

const MEDIAPIPE_HANDS_CONFIG = {'type': 'full', 'render3D': true};
/**
 * This map descripes tunable flags and theior corresponding types.
 *
 * The flags (keys) in the map satisfy the following two conditions:
 * - Is tunable. For example, `IS_BROWSER` and `IS_CHROME` is not tunable,
 * because they are fixed when running the scripts.
 * - Does not depend on other flags when registering in `ENV.registerFlag()`.
 * This rule aims to make the list streamlined, and, since there are
 * dependencies between flags, only modifying an independent flag without
 * modifying its dependents may cause inconsistency.
 * (`WEBGL_RENDER_FLOAT32_CAPABLE` is an exception, because only exposing
 * `WEBGL_FORCE_F16_TEXTURES` may confuse users.)
 */
const TUNABLE_FLAG_VALUE_RANGE_MAP = {
  'WEBGL_VERSION': [1, 2],
  'WASM_HAS_SIMD_SUPPORT': [true, false],
  'WASM_HAS_MULTITHREAD_SUPPORT': [true, false],
  'WEBGL_CPU_FORWARD': [true, false],
  'WEBGL_PACK': [true, false],
  'WEBGL_FORCE_F16_TEXTURES': [true, false],
  'WEBGL_RENDER_FLOAT32_CAPABLE': [true, false],
  'WEBGL_FLUSH_THRESHOLD': [-1, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
  'CHECK_COMPUTATION_FOR_ERRORS': [true, false],
};

const Map<List<String>, List<String>> BACKEND_FLAGS_MAP = {
  ['tfjs-wasm']: ['WASM_HAS_SIMD_SUPPORT', 'WASM_HAS_MULTITHREAD_SUPPORT'],
  ['tfjs-webgl']: [
    'WEBGL_VERSION',
    'WEBGL_CPU_FORWARD',
    'WEBGL_PACK',
    'WEBGL_FORCE_F16_TEXTURES',
    'WEBGL_RENDER_FLOAT32_CAPABLE',
    'WEBGL_FLUSH_THRESHOLD'
  ],
  ['mediapipe-gpu']: []
};

const MODEL_BACKEND_MAP = {
  [handdetection.SupportedModels.mediaPipeHands]: [
    'mediapipe-gpu',
    'tfjs-webgl'
  ]
};

const TUNABLE_FLAG_NAME_MAP = {
  'PROD': 'production mode',
  'WEBGL_VERSION': 'webgl version',
  'WASM_HAS_SIMD_SUPPORT': 'wasm SIMD',
  'WASM_HAS_MULTITHREAD_SUPPORT': 'wasm multithread',
  'WEBGL_CPU_FORWARD': 'cpu forward',
  'WEBGL_PACK': 'webgl pack',
  'WEBGL_FORCE_F16_TEXTURES': 'enforce float16',
  'WEBGL_RENDER_FLOAT32_CAPABLE': 'enable float32',
  'WEBGL_FLUSH_THRESHOLD': 'GL flush wait time(ms)'
};
