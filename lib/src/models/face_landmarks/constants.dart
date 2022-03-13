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

// import * as faceMesh from '@mediapipe/face_mesh';

const MEDIAPIPE_FACEMESH_NUM_KEYPOINTS = 468;
const MEDIAPIPE_FACEMESH_NUM_KEYPOINTS_WITH_IRISES = 478;

List<int> _connectionsToIndices(List<List<int>> connections) {
  final indices = connections.map((connection) => connection[0]).toList();
  indices.add(connections[connections.length - 1][1]);
  return indices;
}

final MEDIAPIPE_KEYPOINTS_BY_CONTOUR = {
  'lips': _connectionsToIndices(faceMesh.FACEMESH_LIPS),
  'leftEye': _connectionsToIndices(faceMesh.FACEMESH_LEFT_EYE),
  'leftEyebrow': _connectionsToIndices(faceMesh.FACEMESH_LEFT_EYEBROW),
  'leftIris': _connectionsToIndices(faceMesh.FACEMESH_LEFT_IRIS),
  'rightEye': _connectionsToIndices(faceMesh.FACEMESH_RIGHT_EYE),
  'rightEyebrow': _connectionsToIndices(faceMesh.FACEMESH_RIGHT_EYEBROW),
  'rightIris': _connectionsToIndices(faceMesh.FACEMESH_RIGHT_IRIS),
  'faceOval': _connectionsToIndices(faceMesh.FACEMESH_FACE_OVAL),
};

const MEDIAPIPE_CONNECTED_KEYPOINTS_PAIRS = faceMesh.FACEMESH_TESSELATION;

/**
 * Maps keypoint index to string label.
 * It turns MEDIAPIPE_KEYPOINTS_BY_CONTOUR which looks like {
 *   lips: [61, 146, 91, 181,...]
 *   leftEye: [263, 249, 390, 373, ...]
 *   ...
 * }
 * to the following form: [
 *   [61, lips],
 *   [146, lips],
 *   ...
 *   [263, leftEye],
 *   [249, leftEye],
 *   ...
 * ]
 */
final MEDIAPIPE_KEYPOINTS = Map<int, String>.fromEntries(
    (MEDIAPIPE_KEYPOINTS_BY_CONTOUR).entries.expand(
          (e) => e.value.map((index) => MapEntry(index, e.key)),
        ));
