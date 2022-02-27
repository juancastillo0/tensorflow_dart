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

// import {nonMaxSuppressionV3Impl} from '../../backends/non_max_suppression_impl';
// import {Tensor1D, Tensor2D} from '../../tensor';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import {nonMaxSuppSanityCheck} from '../nonmax_util';
// import {tensor1d} from '../tensor1d';

import '../_prelude.dart';
import '../tensor.dart';
import 'non_max_util.dart';
import 'image.dart';

/**
 * Performs non maximum suppression of bounding boxes based on
 * iou (intersection over union).
 *
 * This is the async version of `nonMaxSuppression`
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @return A 1D tensor with the selected box indices.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
Future<Tensor1D> nonMaxSuppressionAsync(
    Tensor2D boxes, Tensor1D scores,
    int maxOutputSize, {double? iouThreshold = image.defaultIouThreshold,
    double? scoreThreshold =  image.defaultScoreThreshold,}) {
      
      return execOp('nonMaxSuppressionAsync', () async{
      iouThreshold ??= image.defaultIouThreshold;
      scoreThreshold ??= image.defaultScoreThreshold;

      
  final $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
  final $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');

  final inputs = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold!, scoreThreshold!, null);
  maxOutputSize = inputs.maxOutputSize;
  iouThreshold = inputs.iouThreshold;
  scoreThreshold = inputs.scoreThreshold;

  final boxesAndScores = await Future.wait([$boxes.data(), $scores.data()]);
  final boxesVals = boxesAndScores[0];
  final scoresVals = boxesAndScores[1];

  // We call a cpu based impl directly with the typedarray data  here rather
  // than a kernel because all kernels are synchronous (and thus cannot await
  // .data()).
  final {selectedIndices} = nonMaxSuppressionV3Impl(
      boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
  if ($boxes != boxes) {
    $boxes.dispose();
  }
  if ($scores != scores) {
    $scores.dispose();
  }

  return tensor1d(selectedIndices, 'int32');
  });
}
