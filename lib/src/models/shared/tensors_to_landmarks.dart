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
import 'dart:typed_data';

import 'interfaces/common_interfaces.dart';
import 'interfaces/config_interfaces.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'sigmoid.dart';

// import * as tf from '@tensorflow/tfjs-core';

// import {Keypoint} from './interfaces/common_interfaces';

// import {TensorsToLandmarksConfig} from './interfaces/config_interfaces';
// import {sigmoid} from './sigmoid';

double applyActivation(
    String activation //: 'none'|'sigmoid'
    ,
    double value) {
  return activation == 'none' ? value : sigmoid(value);
}

/**
 * A calculator for converting Tensors from regression models into landmarks.
 * Note that if the landmarks in the tensor has more than 5 dimensions, only the
 * first 5 dimensions will be converted to [x,y,z, visibility, presence]. The
 * latter two fields may also stay unset if such attributes are not supported in
 * the model.
 * @param landmarkTensor List of Tensors of type float32. Only the first tensor
 * will be used. The size of the values must be (num_dimension x num_landmarks).
 * @param flipHorizontally Optional. Whether to flip landmarks horizontally or
 * not. Overrides corresponding field in config.
 * @param flipVertically Optional. Whether to flip landmarks vertically or not.
 * Overrides corresponding field in config.
 *
 * @param config
 *
 * @returns Normalized landmarks.
 */
Future<List<Keypoint>> tensorsToLandmarks(
  tf.Tensor landmarkTensor,
  TensorsToLandmarksConfig config, {
  bool? flipHorizontally,
  bool? flipVertically,
}) async {
  flipHorizontally = flipHorizontally ?? config.flipHorizontally ?? false;
  flipVertically = flipVertically ?? config.flipVertically ?? false;

  final numValues = landmarkTensor.size;
  final numDimensions = numValues ~/ config.numLandmarks;
  final rawLandmarks = await landmarkTensor.data() as Float32List;

  final List<Keypoint> outputLandmarks = [];
  for (int ld = 0; ld < config.numLandmarks; ++ld) {
    final offset = ld * numDimensions;

    double x = 0;
    double y = 0;
    double? z;
    double? score;
    if (flipHorizontally) {
      x = config.inputImageWidth - rawLandmarks[offset];
    } else {
      x = rawLandmarks[offset];
    }
    if (numDimensions > 1) {
      if (flipVertically) {
        y = config.inputImageHeight - rawLandmarks[offset + 1];
      } else {
        y = rawLandmarks[offset + 1];
      }
    }
    if (numDimensions > 2) {
      z = rawLandmarks[offset + 2];
    }
    if (numDimensions > 3) {
      score = applyActivation(
          config.visibilityActivation, rawLandmarks[offset + 3]);
    }
    // presence is in rawLandmarks[offset + 4], we don't expose it.

    outputLandmarks.add(Keypoint(x: x, y: y, z: z, score: score));
  }

  for (int i = 0; i < outputLandmarks.length; ++i) {
    final landmark = outputLandmarks[i];
    final x = landmark.x / config.inputImageWidth;
    final y = landmark.y / config.inputImageHeight;
    // Scale Z coordinate as X + allow additional uniform normalization.
    final z = landmark.z == null
        ? null
        : (landmark.z! / config.inputImageWidth / (config.normalizeZ ?? 1));

    outputLandmarks[i] = landmark.copyWith(
      x: x,
      y: y,
      z: Nullable(z),
    );
  }

  return outputLandmarks;
}
