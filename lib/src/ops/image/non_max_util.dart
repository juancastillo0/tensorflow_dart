/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import {Tensor1D, Tensor2D} from '../tensor';
// import * as util from '../util';

import 'dart:math' as math;
import '../../tensor.dart';
import '../../util_base.dart' as util;

class NonMaxInputs {
  final int maxOutputSize;
  final double iouThreshold;
  final double scoreThreshold;
  final double softNmsSigma;

  NonMaxInputs({
    required this.maxOutputSize,
    required this.iouThreshold,
    required this.scoreThreshold,
    required this.softNmsSigma,
  });
}

NonMaxInputs nonMaxSuppSanityCheck(
  Tensor2D boxes,
  Tensor1D scores,
  int maxOutputSize,
  double iouThreshold,
  double scoreThreshold,
  double? softNmsSigma,
) {
  // TODO:
  // if (iouThreshold == null) {
  //   iouThreshold = 0.5;
  // }
  // if (scoreThreshold == null) {
  //   scoreThreshold = Number.NEGATIVE_INFINITY;
  // }

  softNmsSigma ??= 0.0;

  final numBoxes = boxes.shape[0];
  maxOutputSize = math.min(maxOutputSize, numBoxes);

  util.assert_(0 <= iouThreshold && iouThreshold <= 1,
      () => "iouThreshold must be in [0, 1], but was '${iouThreshold}'");
  util.assert_(boxes.rank == 2,
      () => "boxes must be a 2D tensor, but was of rank '${boxes.rank}'");
  util.assert_(
      boxes.shape[1] == 4,
      () =>
          "boxes must have 4 columns, but 2nd dimension was ${boxes.shape[1]}");
  util.assert_(scores.rank == 1, () => 'scores must be a 1D tensor');
  util.assert_(
      scores.shape[0] == numBoxes,
      () =>
          "scores has incompatible shape with boxes. Expected ${numBoxes}, " +
          "but was ${scores.shape[0]}");
  util.assert_(0 <= softNmsSigma && softNmsSigma <= 1,
      () => "softNmsSigma must be in [0, 1], but was '${softNmsSigma}'");

  return NonMaxInputs(
    maxOutputSize: maxOutputSize,
    iouThreshold: iouThreshold,
    scoreThreshold: scoreThreshold,
    softNmsSigma: softNmsSigma,
  );
}
