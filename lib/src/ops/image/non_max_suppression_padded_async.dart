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
// import {nonMaxSuppressionV4Impl} from '../../backends/non_max_suppression_impl';
// import {Tensor1D, Tensor2D} from '../../tensor';
// import {NamedTensorMap} from '../../tensor_types';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import {nonMaxSuppSanityCheck} from '../nonmax_util';
// import {scalar} from '../scalar';
// import {tensor1d} from '../tensor1d';

import 'package:tensorflow_wasm/src/backends/non_max_suppression_impl.dart';

import 'image.dart';
import '../_prelude.dart';
import '../scalar.dart';
import '../tensor.dart';
import 'non_max_util.dart';

/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
 * iou (intersection over union), with an option to pad results.
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
 * @param padToMaxOutputSize Defalts to false. If true, size of output
 *     `selectedIndices` is padded to maxOutputSize.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - validOutputs: A scalar denoting how many elements in `selectedIndices`
 *       are valid. Valid elements occur first, then padding.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
Future<NmsPadded> nonMaxSuppressionPaddedAsync(
  Tensor2D boxes,
  Tensor1D scores,
  int maxOutputSize, {
  double? iouThreshold = image.defaultIouThreshold,
  double? scoreThreshold = image.defaultScoreThreshold,
  bool padToMaxOutputSize = false,
}) async {
  iouThreshold ??= image.defaultIouThreshold;
  scoreThreshold ??= image.defaultScoreThreshold;
  final $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
  final $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');

  final params = nonMaxSuppSanityCheck($boxes, $scores, maxOutputSize,
      iouThreshold, scoreThreshold, null /* softNmsSigma */);
  final $maxOutputSize = params.maxOutputSize;
  final $iouThreshold = params.iouThreshold;
  final $scoreThreshold = params.scoreThreshold;

  final vals = await Future.wait([$boxes.data(), $scores.data()]);

  // We call a cpu based impl directly with the typedarray data here rather
  // than a kernel because all kernels are synchronous (and thus cannot await
  // .data()).
  final _r = nonMaxSuppressionV4Impl(vals[0].cast(), vals[1].cast(),
      $maxOutputSize, $iouThreshold, $scoreThreshold, padToMaxOutputSize);
  final selectedIndices = _r.selectedIndices;
  final validOutputs = _r.validOutputs!;

  if ($boxes != boxes) {
    $boxes.dispose();
  }
  if ($scores != scores) {
    $scores.dispose();
  }

  return NmsPadded(
    selectedIndices: tensor1d(selectedIndices, 'int32'),
    validOutputs: scalar(validOutputs, 'int32'),
  );
}
