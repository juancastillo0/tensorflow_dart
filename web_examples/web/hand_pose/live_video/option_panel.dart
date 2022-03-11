// /**
//  * @license
//  * Copyright 2021 Google LLC. All Rights Reserved.
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * https://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  * =============================================================================
//  */
// // import * as handdetection from '@tensorflow-models/hand-pose-detection';
// // import * as tf from '@tensorflow/tfjs-core';

// // import * as params from './shared/params';

// import 'dart:html';

// import '../shared/params.dart' as params;
// import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
// import 'package:tensorflow_wasm/models/hand_pose.dart' as handdetection;

// /**
//  * Records each flag's default value under the runtime environment and is a
//  * constant in runtime.
//  */
// Map<String, Object?> TUNABLE_FLAG_DEFAULT_VALUE_MAP = {};

// final stringValueMap = {};

// setupDatGui(UrlSearchParams urlParams) async {
//   final gui = dat.GUI({width: 300});
//   gui.domElement.id = 'gui';

//   // The camera folder contains options for video settings.
//   final cameraFolder = gui.addFolder('Camera');
//   final fpsController = cameraFolder.add(params.STATE.camera, 'targetFPS');
//   fpsController.onFinishChange((_)  {
//     params.STATE.isTargetFPSChanged = true;
//   });
//   final sizeController = cameraFolder.add(
//       params.STATE.camera, 'sizeOption', params.VIDEO_SIZE.keys);
//   sizeController.onChange((_) {
//     params.STATE.isSizeOptionChanged = true;
//   });
//   cameraFolder.open();

//   // The model folder contains options for model selection.
//   final modelFolder = gui.addFolder('Model');

//   final model = urlParams.get('model');
//   var type = urlParams.get('type');
//   var maxNumHands = int.tryParse(urlParams.get('maxNumHands') ?? '');

//   switch (model) {
//     case 'mediapipe_hands':
//       params.STATE.model = handdetection.SupportedModels.mediaPipeHands;
//       if (type != 'full' && type != 'lite') {
//         // Nulify invalid value.
//         type = null;
//       }
//       if (maxNumHands == null || maxNumHands < 1 ) {
//         // Nulify invalid value.
//         maxNumHands = 2;
//       }
//       break;
//     default:
//       window.alert('${urlParams.get('model')}');
//       break;
//   }

//   final modelController = modelFolder.add(
//       params.STATE, 'model', handdetection.SupportedModels.values);

//   modelController.onChange((_) {
//     params.STATE.isModelChanged = true;
//     _showModelConfigs(modelFolder);
//     _showBackendConfigs(backendFolder);
//   });

//   _showModelConfigs(modelFolder, type, maxNumHands);

//   modelFolder.open();

//   final backendFolder = gui.addFolder('Backend');

//   _showBackendConfigs(backendFolder);

//   backendFolder.open();

//   return gui;
// }

// _showBackendConfigs(folderController) async {
//   // Clean up backend configs for the previous model.
//   final fixedSelectionCount = 0;
//   while (folderController.__controllers.length > fixedSelectionCount) {
//     folderController.remove(
//         folderController
//             .__controllers[folderController.__controllers.length - 1]);
//   }
//   final backends = params.MODEL_BACKEND_MAP[params.STATE.model]!;
//   // The first element of the array is the default backend for the model.
//   params.STATE.backend = backends[0];
//   final backendController =
//       folderController.add(params.STATE, 'backend', backends);
//   backendController.name('runtime-backend');
//   backendController.onChange((backend) async  {
//     params.STATE.isBackendChanged = true;
//     await _showFlagSettings(folderController, backend);
//   });
//   await _showFlagSettings(folderController, params.STATE.backend);
// }

// _showModelConfigs(folderController, [String? type, int? maxNumHands]) {
//   // Clean up model configs for the previous model.
//   // The first constroller under the `folderController` is the model
//   // selection.
//   final fixedSelectionCount = 1;
//   while (folderController.__controllers.length > fixedSelectionCount) {
//     folderController.remove(
//         folderController
//             .__controllers[folderController.__controllers.length - 1]);
//   }

//   switch (params.STATE.model) {
//     case handdetection.SupportedModels.mediaPipeHands:
//       _addMediaPipeHandsControllers(folderController, type, maxNumHands);
//       break;
//     default:
//       window.alert('Model ${params.STATE.model} is not supported.');
//   }
// }

// // The MediaPipeHands model config folder contains options for MediaPipeHands config
// // settings.
// _addMediaPipeHandsControllers(modelConfigFolder, String? type, int? maxNumHands) {
//   params.STATE.modelConfig = {...params.MEDIAPIPE_HANDS_CONFIG};
//   params.STATE.modelConfig.type = type ?? 'full';
//   params.STATE.modelConfig.maxNumHands = maxNumHands ?? 2;

//   final typeController = modelConfigFolder.add(
//       params.STATE.modelConfig, 'type', ['lite', 'full']);
//   typeController.onChange((_) {
//     // Set isModelChanged to true, so that we don't render any result during
//     // changing models.
//     params.STATE.isModelChanged = true;
//   });

//   final maxNumHandsController = modelConfigFolder.add(
//     params.STATE.modelConfig, 'maxNumHands', 1, 10).step(1);
//     maxNumHandsController.onChange((_) {
//     // Set isModelChanged to true, so that we don't render any result during
//     // changing models.
//     params.STATE.isModelChanged = true;
//   });

//   final render3DController =
//       modelConfigFolder.add(params.STATE.modelConfig, 'render3D');
//   render3DController.onChange((render3D) {
//     document.querySelector('#scatter-gl-container-left')?.style.display =
//         render3D ? 'inline-block' : 'none';
//     document.querySelector('#scatter-gl-container-right')?.style.display =
//         render3D ? 'inline-block' : 'none';
//   });
// }

// /**
//  * Query all tunable flags' default value and populate `STATE.flags` with them.
//  */
// Future<void> _initDefaultValueMap() async {
//   // Clean up the cache to query tunable flags' default values.
//   TUNABLE_FLAG_DEFAULT_VALUE_MAP = {};
//   params.STATE.flags = {};
//   for (final backendFlags in params.BACKEND_FLAGS_MAP.values) {
//     for (int index = 0; index < backendFlags.length;
//          index++) {
//       final flag = backendFlags[index];
//       TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag] = await tf.env().getAsync(flag);
//     }
//   }

//   // Initialize STATE.flags with tunable flags' default values.
//   for (final flag in TUNABLE_FLAG_DEFAULT_VALUE_MAP) {
//     if (params.BACKEND_FLAGS_MAP[params.STATE.backend]!.contains(flag)) {
//       params.STATE.flags[flag] = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
//     }
//   }
// }

// /**
//  * Heuristically determine flag's value range based on flag's default value.
//  *
//  * Assume that the flag's default value has already chosen the best option for
//  * the runtime environment, so users can only tune the flag value downwards.
//  *
//  * For example, if the default value of `WEBGL_RENDER_FLOAT32_CAPABLE` is false,
//  * the tunable range is [false]; otherwise, the tunable range is [true. false].
//  *
//  * @param {string} flag
//  */
//  _getTunableRange(String flag) {
//   final defaultValue = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
//   if (flag == 'WEBGL_FORCE_F16_TEXTURES') {
//     return [false, true];
//   } else if (flag == 'WEBGL_VERSION') {
//     final tunableRange = <int>[];
//     for (int value = 1; value <= (defaultValue as num); value++) {
//       tunableRange.add(value);
//     }
//     return tunableRange;
//   } else if (flag == 'WEBGL_FLUSH_THRESHOLD') {
//     final tunableRange = <double>[-1];
//     for (double value = 0; value <= 2; value += 0.25) {
//       tunableRange.add(value);
//     }
//     return tunableRange;
//   } else if (defaultValue is bool) {
//     return defaultValue ? [false, true] : [false];
//   } else if (params.TUNABLE_FLAG_VALUE_RANGE_MAP[flag] != null) {
//     return params.TUNABLE_FLAG_VALUE_RANGE_MAP[flag];
//   } else {
//     return [defaultValue];
//   }
// }

// /**
//  * Show flag settings for the given backend under the UI element of
//  * `folderController`.
//  *
//  * @param {dat.gui.GUI} folderController
//  * @param {string} backendName
//  */
// _showBackendFlagSettings(folderController, String backendName) {
//   final tunableFlags = params.BACKEND_FLAGS_MAP[backendName]!;
//   for (int index = 0; index < tunableFlags.length; index++) {
//     final flag = tunableFlags[index];
//     final flagName = params.TUNABLE_FLAG_NAME_MAP[flag] ?? flag;

//     // When tunable (bool) and range (array) attributes of `flagRegistry` is
//     // implemented, we can apply them to here.
//     final flagValueRange = _getTunableRange(flag);
//     // Heuristically consider a flag with at least two options as tunable.
//     if (flagValueRange.length < 2) {
//       console.warn(
//           'The ${flag} is considered as untunable, ' +
//           'because its value range is [${flagValueRange}].');
//       continue;
//     }

//     let flagController;
//     if (typeof flagValueRange[0] == 'boolean') {
//       // Show checkbox for boolean flags.
//       flagController = folderController.add(params.STATE.flags, flag);
//     } else {
//       // Show dropdown for other types of flags.
//       flagController =
//           folderController.add(params.STATE.flags, flag, flagValueRange);

//       // Because dat.gui always casts dropdown option values to string, we need
//       // `stringValueMap` and `onFinishChange()` to recover the value type.
//       if (stringValueMap[flag] == null) {
//         stringValueMap[flag] = {};
//         for (int index = 0; index < flagValueRange.length; index++) {
//           final realValue = flagValueRange[index];
//           final stringValue = String(flagValueRange[index]);
//           stringValueMap[flag][stringValue] = realValue;
//         }
//       }
//       flagController.onFinishChange((stringValue) {
//         params.STATE.flags[flag] = stringValueMap[flag][stringValue];
//       });
//     }
//     flagController.name(flagName).onChange(() {
//       params.STATE.isFlagChanged = true;
//     });
//   }
// }

// /**
//  * Set up flag settings under the UI element of `folderController`:
//  * - If it is the first call, initialize the flags' default value and show flag
//  * settings for both the general and the given backend.
//  * - Else, clean up flag settings for the previous backend and show flag
//  * settings for the new backend.
//  *
//  * @param {dat.gui.GUI} folderController
//  * @param {string} backendName
//  */
// _showFlagSettings(folderController, String backendName) async {
//   await _initDefaultValueMap();

//   // Clean up flag settings for the previous backend.
//   // The first constroller under the `folderController` is the backend
//   // setting.
//   final fixedSelectionCount = 1;
//   while (folderController.__controllers.length > fixedSelectionCount) {
//     folderController.remove(
//         folderController
//             .__controllers[folderController.__controllers.length - 1]);
//   }

//   // Show flag settings for the new backend.
//   _showBackendFlagSettings(folderController, backendName);
// }