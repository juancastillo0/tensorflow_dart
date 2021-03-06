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
// import {transformValueRange} from './image_utils';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'image_utils.dart';

tf.Tensor4D shiftImageValue(tf.Tensor4D image, List<double> outputFloatRange) {
  // Calculate the scale and offset to shift from [0, 255] to [-1, 1].
  final valueRange = transformValueRange(
      0, 255, outputFloatRange[0] /* min */, outputFloatRange[1] /* max */);

  // Shift value range.
  return tf.tidy(
    () => tf.add(
      tf.mul(image, tf.scalar(valueRange.scale)),
      tf.scalar(valueRange.offset),
    ),
  );
}
