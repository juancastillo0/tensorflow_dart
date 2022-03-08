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
// import * as tf from '@tensorflow/tfjs-core';
// import {splitDetectionResult} from './split_detection_result';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'split_detection_result.dart';

class DetectorResult {
  final tf.Tensor2D boxes;
  final tf.Tensor1D logits;

  const DetectorResult({
    required this.boxes,
    required this.logits,
  });
}

DetectorResult detectorResult(tf.Tensor3D detectionResult) {
  return tf.tidy(() {
    final _v = splitDetectionResult(detectionResult);
    final logits = _v.logits;
    final rawBoxes = _v.rawBoxes;
    // Shape [896, 12]
    final rawBoxes2d = tf.squeeze(rawBoxes);
    // Shape [896]
    final logits1d = tf.squeeze(logits);

    return DetectorResult(
      boxes: rawBoxes2d as tf.Tensor2D,
      logits: logits1d as tf.Tensor1D,
    );
  });
}
