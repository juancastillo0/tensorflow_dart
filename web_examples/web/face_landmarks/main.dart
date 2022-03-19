/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
// import '@tensorflow/tfjs-backend-cpu';

// import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import * as tf from '@tensorflow/tfjs-core';
// import Stats from 'stats.js';

// import {TRIANGULATION} from './triangulation';

import 'dart:async';
import 'dart:math' as Math;
import 'package:mobx/mobx.dart';

import 'triangulation.dart';
import 'package:universal_html/html.dart';
import 'package:tensorflow_wasm/models/face_landmarks.dart'
    as faceLandmarksDetection;

import 'ui.dart';

const CANVAS_ELEMENT_ID = 'mesh-canvas';
const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = '#32EEDB';
const RED = '#FF2C35';
const BLUE = '#157AB3';
bool stopRendering = false;

bool _isMobile() {
  final isAndroid = RegExp('Android', caseSensitive: false)
      .hasMatch(window.navigator.userAgent);
  final isiOS = RegExp('iPhone|iPad|iPod', caseSensitive: false)
      .hasMatch(window.navigator.userAgent);
  return isAndroid || isiOS;
}

double _distance(
    faceLandmarksDetection.Keypoint a, faceLandmarksDetection.Keypoint b) {
  return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
}

void _drawPath(ctx, points, closePath) {
  final region = Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (int i = 1; i < points.length; i++) {
    final point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

faceLandmarksDetection.FaceLandmarksDetector? model;
late CanvasRenderingContext2D ctx;
int? videoWidth;
int? videoHeight;
late VideoElement video;
late CanvasElement _canvas;
// scatterGLHasInitialized = false,
// scatterGL,
int? rafID;

const VIDEO_SIZE = 500;
final mobile = _isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
final renderPointcloud = mobile == false;
// const stats = Stats();
final state = FaceLandmarkState();

final _modelCompleter = Completer();

class FaceLandmarkState {
  final backend = Observable('wasm');
  final maxFaces = Observable(1);
  final triangulateMesh = Observable(true);
  final predictIrises = Observable(true);
}

Future<void> _setupDatGui() async {
  // final gui = dat.GUI();
  // gui.add(state, 'backend', ['webgl', 'wasm', 'cpu'])
  //     .onChange((backend) async {
  //       stopRendering = true;
  //       window.cancelAnimationFrame(rafID);
  //       await tf.setBackend(backend);
  //       stopRendering = false;
  //       requestAnimationFrame(renderPrediction);
  //     });

  await setupFaceLandmarkGui(state);

  autorun((_) async {
    model = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      faceLandmarksDetection.MediaPipeFaceMeshTfjsModelConfig(
        maxFaces: state.maxFaces.value,
        refineLandmarks: state.predictIrises.value,
      ),
    );
    if (!_modelCompleter.isCompleted) {
      _modelCompleter.complete();
    }
  });

  // if (renderPointcloud) {
  //   gui.add(state, 'renderPointcloud').onChange((render) {
  //     document.querySelector('#scatter-gl-container').style.display =
  //         render ? 'inline-block' : 'none';
  //   });
  // }
}

Future<VideoElement> _setupCamera() async {
  video = document.getElementById('video') as VideoElement;

  final stream = await window.navigator.mediaDevices!.getUserMedia({
    'audio': false,
    'video': {
      'facingMode': 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      'width': mobile ? null : VIDEO_SIZE,
      'height': mobile ? null : VIDEO_SIZE
    },
  });
  video.srcObject = stream;

  return video.onLoadedMetadata.first.then((value) => video);
}

Future<void> _renderPrediction([_]) async {
  if (stopRendering) {
    return;
  }

  // stats.begin();

  final predictions = await model!.estimateFaces(
    video,
    faceLandmarksDetection.MediaPipeFaceMeshTfjsEstimationConfig(
      // returnTensors: false,
      flipHorizontal: false,
      // predictIrises: state.predictIrises,
    ),
  );
  ctx.drawImageScaledFromSource(
    video,
    0,
    0,
    videoWidth!,
    videoHeight!,
    0,
    0,
    _canvas.width!,
    _canvas.height!,
  );

  if (predictions.length > 0) {
    predictions.forEach((prediction) {
      final keypoints = prediction.keypoints;

      if (state.triangulateMesh.value) {
        ctx.strokeStyle = GREEN;
        ctx.lineWidth = 0.5;

        for (int i = 0; i < TRIANGULATION.length / 3; i++) {
          final points = [
            TRIANGULATION[i * 3],
            TRIANGULATION[i * 3 + 1],
            TRIANGULATION[i * 3 + 2]
          ].map((index) => keypoints[index]);

          _drawPath(ctx, points, true);
        }
      } else {
        ctx.fillStyle = GREEN;

        for (int i = 0; i < NUM_KEYPOINTS; i++) {
          final x = keypoints[i].x;
          final y = keypoints[i].y;

          ctx.beginPath();
          ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.pi);
          ctx.fill();
        }
      }

      if (keypoints.length > NUM_KEYPOINTS) {
        ctx.strokeStyle = RED;
        ctx.lineWidth = 1;

        final leftCenter = keypoints[NUM_KEYPOINTS];
        final leftDiameterY = _distance(
            keypoints[NUM_KEYPOINTS + 4], keypoints[NUM_KEYPOINTS + 2]);
        final leftDiameterX = _distance(
            keypoints[NUM_KEYPOINTS + 3], keypoints[NUM_KEYPOINTS + 1]);

        ctx.beginPath();
        ctx.ellipse(leftCenter.x, leftCenter.y, leftDiameterX / 2,
            leftDiameterY / 2, 0, 0, 2 * Math.pi, null);
        ctx.stroke();

        if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
          final rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
          final rightDiameterY = _distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
          final rightDiameterX = _distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);

          ctx.beginPath();
          ctx.ellipse(rightCenter.x, rightCenter.y, rightDiameterX / 2,
              rightDiameterY / 2, 0, 0, 2 * Math.pi, null);
          ctx.stroke();
        }
      }
    });

    // if (renderPointcloud && state.renderPointcloud && scatterGL != null) {
    //   final pointsData = predictions.map((prediction) {
    //     let scaledMesh = prediction.scaledMesh;
    //     return scaledMesh.map((point) => ([-point[0], -point[1], -point[2]]));
    //   });

    //   List<double> flattenedPointsData = [];
    //   for (int i = 0; i < pointsData.length; i++) {
    //     flattenedPointsData = flattenedPointsData.concat(pointsData[i]);
    //   }
    //   final dataset = ScatterGL.Dataset(flattenedPointsData);

    //   if (!scatterGLHasInitialized) {
    //     scatterGL.setPointColorer((i) {
    //       if (i % (NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS * 2) > NUM_KEYPOINTS) {
    //         return RED;
    //       }
    //       return BLUE;
    //     });
    //     scatterGL.render(dataset);
    //   } else {
    //     scatterGL.updateDataset(dataset);
    //   }
    //   scatterGLHasInitialized = true;
    // }
  }

  // stats.end();
  rafID = window.requestAnimationFrame(_renderPrediction);
}

Future<void> main() async {
  // tfjsWasm.setWasmPaths(
  //     'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.versionWasm}/dist/');
  // if (renderPointcloud) {
  //   state.renderPointcloud = true;
  // }

  await _setupDatGui();
  await Future.wait([
    // tf.setBackend(tfjsWasm.wasmBackendFactory),
    _setupCamera(),
  ]);

  // stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  // document.getElementById('main')!.children.add(stats.dom);
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth!;
  video.height = videoHeight!;

  _canvas = document.getElementById(CANVAS_ELEMENT_ID) as CanvasElement;
  _canvas.width = videoWidth;
  _canvas.height = videoHeight;
  final canvasContainer = document.querySelector('.canvas-wrapper')!;
  canvasContainer.setAttribute(
      'style', 'width: ${videoWidth}px; height: ${videoHeight}px');

  ctx = _canvas.getContext('2d') as CanvasRenderingContext2D;
  ctx.translate(_canvas.width!, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = GREEN;
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 0.5;

  await _modelCompleter.future;
  _renderPrediction();

  // if (renderPointcloud) {
  //   document.querySelector('#scatter-gl-container').style =
  //       'width: ${VIDEO_SIZE}px; height: ${VIDEO_SIZE}px;';

  //   scatterGL = ScatterGL(document.querySelector('#scatter-gl-container'),
  //       {'rotateOnStart': false, 'selectEnabled': false});
  // }
}
