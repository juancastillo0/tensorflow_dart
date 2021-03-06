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
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

class SplitDetectionResult {
  final tf.Tensor3D logits;
  final tf.Tensor3D rawBoxes;

  SplitDetectionResult({
    required this.logits,
    required this.rawBoxes,
  });
}

SplitDetectionResult splitDetectionResult(tf.Tensor3D detectionResult) {
  return tf.tidy(() {
    // logit is stored in the first element in each anchor data.
    final logits = tf.slice(detectionResult, [0, 0, 0], [1, -1, 1]);
    // Bounding box coords are stored in the next four elements for each anchor
    // point.
    final rawBoxes = tf.slice(detectionResult, [0, 0, 1], [1, -1, -1]);

    return SplitDetectionResult(logits: logits, rawBoxes: rawBoxes);
  });
}
