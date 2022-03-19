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

// import '@tensorflow/tfjs-backend-webgl';
// import * as mpHands from '@mediapipe/hands';

// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import * as handdetection from '@tensorflow-models/hand-pose-detection';

import 'dart:html';

import 'package:tensorflow_wasm/models/hand_pose.dart' as handdetection;
import 'package:tensorflow_wasm/src/models/hand_pose/single/hand_pose_single.dart'
    as handdsingle;
import '../shared/params.dart';
import '../shared/util.dart';
import 'camera.dart';
import 'option_panel_deact.dart';

// import {Camera} from './camera';
// import {setupDatGui} from './option_panel';
// import {STATE} from './shared/params';
// import {setupStats} from './shared/stats_panel';
// import {setBackendAndEnvFlags} from './shared/util';

handdetection.HandDetector? detector;
Camera? camera;
var stats;
int startInferenceTime = 0, numInferences = 0;
int inferenceTimeSum = 0, lastPanelUpdate = 0;
var rafId;

void alert(Object message) {
  window.alert(message.toString());
}

Future<handdetection.HandDetector> _createDetector() async {
  switch (STATE.model) {
    case handdetection.SupportedModels.mediaPipeHands:
      final runtime = STATE.backend.split('-')[0];
      // if (runtime === 'mediapipe') {
      //   return handdetection._createDetector(STATE.model, {
      //     runtime,
      //     modelType: STATE.modelConfig.type,
      //     maxHands: STATE.modelConfig.maxNumHands,
      //     solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}`
      //   });
      // } else if (runtime === 'tfjs') {
      return handdsingle.load(handdetection.MediaPipeHandsTfjsModelConfig(
        // runtime,
        modelType: STATE.modelConfig.type,
        maxHands: STATE.modelConfig.maxNumHands,
      ));
    // return handdetection.createDetector(
    //   STATE.model,
    //   handdetection.MediaPipeHandsTfjsModelConfig(
    //     // runtime,
    //     modelType: STATE.modelConfig.type,
    //     maxHands: STATE.modelConfig.maxNumHands,
    //   ),
    // );
    // }
  }
}

Future<void> _checkGuiUpdate() async {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    detector?.dispose();

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await _createDetector();
    } catch (error, s) {
      detector = null;
      print('$error $s');
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

void _beginEstimateHandsStats() {
  startInferenceTime = DateTime.now().millisecondsSinceEpoch;
}

void _endEstimateHandsStats() {
  final endInferenceTime = DateTime.now().millisecondsSinceEpoch;
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  final panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    final averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    // TODO:
    // stats.customFpsPanel
    //     .update(1000.0 / averageInferenceTime, 120 /* maxValue */);
    // lastPanelUpdate = endInferenceTime;
  }
}

Future<void> renderResult() async {
  if (camera!.video.readyState < 2) {
    await camera!.video.onLoadedData.first;
  }

  List<handdetection.Hand>? hands = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateHands.
    _beginEstimateHandsStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      hands = await detector!.estimateHands(
        camera!.video,
        handdetection.EstimationConfig(flipHorizontal: false),
      );
    } catch (error, s) {
      detector!.dispose();
      detector = null;
      print('$error $s');
    }

    _endEstimateHandsStats();
  }

  camera!.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (hands != null && hands.length > 0 && !STATE.isModelChanged) {
    camera!.drawResults(hands);
  }
}

Future<void> renderPrediction([_]) async {
  await _checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = window.requestAnimationFrame(renderPrediction);
}

Future<void> app() async {
  // Gui content will change depending on which model is in the query string.
  final urlParams = UrlSearchParams(window.location.search);

  final model = urlParams.get('model');
  if (model == null) {
    alert('Cannot find model in the query string.');
    return;
  }

  var type = urlParams.get('type');
  var maxNumHands = int.tryParse(urlParams.get('maxNumHands') ?? '');

  switch (model) {
    case 'mediapipe_hands':
      // STATE.model = handdetection.SupportedModels.mediaPipeHands;
      if (type != 'full' && type != 'lite') {
        // Nulify invalid value.
        type = null;
      }
      if (maxNumHands == null || maxNumHands < 1) {
        // Nulify invalid value.
        maxNumHands = 2;
      }
      break;
    // default:
    //   window.alert('${urlParams.get('model')}');
    //   break;
  }
  if (maxNumHands != null) {
    STATE.modelConfig.maxNumHands = maxNumHands;
  }

  await setupDatGui(urlParams);

  // TODO:
  // stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await _createDetector();

  renderPrediction();
}
