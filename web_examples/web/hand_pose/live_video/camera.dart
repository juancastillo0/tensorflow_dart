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
// import * as scatter from 'scatter-gl';

// import * as params from './shared/params';
// import {isMobile} from './shared/util';

import 'dart:html';
import 'dart:math' as Math;
import 'package:tensorflow_wasm/models/hand_pose.dart'
    show Hand, Handedness, Keypoint;

import '../shared/params.dart' as params;
import '../shared/util.dart' show isMobile;

// These anchor points allow the hand pointcloud to resize according to its
// position in the input.
const ANCHOR_POINTS = [
  [0, 0, 0],
  [0, 0.1, 0],
  [-0.1, 0, 0],
  [-0.1, -0.1, 0]
];

const fingerLookupIndices = {
  'thumb': [0, 1, 2, 3, 4],
  'indexFinger': [0, 5, 6, 7, 8],
  'middleFinger': [0, 9, 10, 11, 12],
  'ringFinger': [0, 13, 14, 15, 16],
  'pinky': [0, 17, 18, 19, 20],
}; // for rendering each finger as a polyline

const connections = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20]
];

// function createScatterGLContext(String selectors) {
//   final scatterGLEl = document.querySelector(selectors);
//   return {
//     scatterGLEl,
//     scatterGL: scatter.ScatterGL(scatterGLEl, {
//       'rotateOnStart': true,
//       'selectEnabled': false,
//       'styles': {polyline: {defaultOpacity: 1, deselectedOpacity: 1}}
//     }),
//     scatterGLHasInitialized: false,
//   };
// }

// final scatterGLCtxtLeftHand = createScatterGLContext('#scatter-gl-container-left');
// final scatterGLCtxtRightHand = createScatterGLContext('#scatter-gl-container-right');

class Camera {
  Camera()
      : this.video = document.getElementById('video') as VideoElement,
        this.canvas =
            document.getElementById(params.CANVAS_ELEMENT_ID) as CanvasElement,
        this.ctx =
            (document.getElementById(params.CANVAS_ELEMENT_ID) as CanvasElement)
                .getContext('2d') as CanvasRenderingContext2D;

  final VideoElement video;
  final CanvasElement canvas;
  final CanvasRenderingContext2D ctx;

  /**
   * Initiate a Camera instance and wait for the camera stream to be ready.
   * @param cameraParam From app `STATE.camera`.
   */
  static setupCamera(params.CameraParams p) async {
    if (window.navigator.mediaDevices == null) {
      throw Exception(
          'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    final $size = params.VIDEO_SIZE[p.sizeOption.value];
    final videoConfig = {
      'audio': false,
      'video': {
        'facingMode': 'user',
        // Only setting the video to a specified size for large screen, on
        // mobile devices accept the default size.
        'width': isMobile()
            ? params.VIDEO_SIZE['360 X 270']!['width']
            : $size!['width'],
        'height': isMobile()
            ? params.VIDEO_SIZE['360 X 270']!['height']
            : $size!['height'],
        'frameRate': {
          'ideal': p.targetFPS.value,
        }
      }
    };

    final MediaStream stream;
    try {
      stream = await window.navigator.mediaDevices!.getUserMedia(videoConfig);
    } catch (e, s) {
      print('$e, $s');
      rethrow;
    }

    final camera = Camera();
    camera.video.srcObject = stream;

    await camera.video.onLoadedMetadata.first;

    camera.video.play();

    final videoWidth = camera.video.videoWidth;
    final videoHeight = camera.video.videoHeight;
    // Must set below two lines, otherwise video element doesn't show.
    camera.video.width = videoWidth;
    camera.video.height = videoHeight;

    camera.canvas.width = videoWidth;
    camera.canvas.height = videoHeight;
    final canvasContainer = document.querySelector('.canvas-wrapper')!;
    canvasContainer.setAttribute(
        'style', 'width: ${videoWidth}px; height: ${videoHeight}px');

    // Because the image from camera is mirrored, need to flip horizontally.
    camera.ctx.translate(camera.video.videoWidth, 0);
    camera.ctx.scale(-1, 1);

    // for (final ctxt in [scatterGLCtxtLeftHand, scatterGLCtxtRightHand]) {
    //   ctxt.scatterGLEl.style =
    //       'width: ${videoWidth / 2}px; height: ${videoHeight / 2}px;';
    //   ctxt.scatterGL.resize();

    //   ctxt.scatterGLEl.style.display =
    //       params.STATE.modelConfig.render3D ? 'inline-block' : 'none';
    // }

    return camera;
  }

  drawCtx() {
    this.ctx.drawImageScaled(
        this.video, 0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  /**
   * Draw the keypoints on the video.
   * @param hands A list of hands to render.
   */
  drawResults(List<Hand> hands) {
    // Sort by right to left hands.
    hands.sort((hand1, hand2) {
      if (hand1.handedness.index < hand2.handedness.index) return 1;
      if (hand1.handedness.index > hand2.handedness.index) return -1;
      return 0;
    });

    // Pad hands to clear empty scatter GL plots.
    // while (hands.length < 2) hands.add({});

    for (int i = 0; i < hands.length; ++i) {
      // Third hand and onwards scatterGL context is set to null since we
      // don't render them.
      final ctxt = null; // [scatterGLCtxtLeftHand, scatterGLCtxtRightHand][i];
      this.drawResult(hands[i], ctxt);
    }
  }

  /**
   * Draw the keypoints on the video.
   * @param hand A hand with keypoints to render.
   * @param ctxt Scatter GL context to render 3D keypoints to.
   */
  drawResult(Hand hand, ctxt) {
    if (hand.keypoints != null) {
      this.drawKeypoints(hand.keypoints, hand.handedness);
    }
    // // Don't render 3D hands after first two.
    // if (ctxt == null) {
    //   return;
    // }
    // if (hand.keypoints3D != null && params.STATE.modelConfig.render3D) {
    //   this.drawKeypoints3D(hand.keypoints3D, hand.handedness, ctxt);
    // } else {
    //   // Clear scatter plot.
    //   this.drawKeypoints3D([], '', ctxt);
    // }
  }

  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints.
   * @param handedness Label of hand (either Left or Right).
   */
  drawKeypoints(List<Keypoint> keypoints, Handedness handedness) {
    final keypointsArray = keypoints;
    this.ctx.fillStyle = handedness == Handedness.left ? 'Red' : 'Blue';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    for (int i = 0; i < keypointsArray.length; i++) {
      final y = keypointsArray[i].x;
      final x = keypointsArray[i].y;
      this.drawPoint(x - 2, y - 2, 3);
    }

    for (final indices in fingerLookupIndices.values) {
      final points = indices.map((idx) => keypoints[idx]).toList();
      this.drawPath(points, false);
    }
  }

  drawPath(List<Keypoint> points, bool closePath) {
    final region = new Path2D();
    region.moveTo(points[0].x, points[0].y);
    for (int i = 1; i < points.length; i++) {
      final point = points[i];
      region.lineTo(point.x, point.y);
    }

    if (closePath) {
      region.closePath();
    }
    this.ctx.stroke(region);
  }

  drawPoint(num y, num x, num r) {
    this.ctx.beginPath();
    this.ctx.arc(x, y, r, 0, 2 * Math.pi);
    this.ctx.fill();
  }

  // drawKeypoints3D(keypoints, handedness, ctxt) {
  //   final scoreThreshold = params.STATE.modelConfig.scoreThreshold ?? 0;
  //   final pointsData =
  //       keypoints.map((keypoint) => ([-keypoint.x, -keypoint.y, -keypoint.z]));

  //   final dataset =
  //       new scatter.ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);

  //   ctxt.scatterGL.setPointColorer((i) {
  //     if (keypoints[i] == null || keypoints[i].score < scoreThreshold) {
  //       // hide anchor points and low-confident points.
  //       return '#ffffff';
  //     }
  //     return handedness == 'Left' ? '#ff0000' : '#0000ff';
  //   });

  //   if (!ctxt.scatterGLHasInitialized) {
  //     ctxt.scatterGL.render(dataset);
  //   } else {
  //     ctxt.scatterGL.updateDataset(dataset);
  //   }
  //   final sequences = connections.map((pair) => ({indices: pair}));
  //   ctxt.scatterGL.setSequences(sequences);
  //   ctxt.scatterGLHasInitialized = true;
  // }
}
