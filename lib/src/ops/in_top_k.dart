/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// import {Tensor} from '../tensor';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';
// import {assert, assertShapesMatch, getTypedArrayFromDType} from '../util';
// import {tensor} from './tensor';

// import 'package:tensorflow_wasm/tensorflow_wasm.dart';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

import '../util_base.dart';
import '_prelude.dart';

/**
 * Returns whether the targets are in the top K predictions.
 *
 * ```js
 * const predictions = tf.tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
 * const targets = tf.tensor1d([2, 0]);
 * const precision = await tf.inTopKAsync(predictions, targets);
 * precision.print();
 * ```
 * @param predictions 2-D or higher `tf.Tensor` with last dimension being
 *     at least `k`.
 * @param targets 1-D or higher `tf.Tensor`.
 * @param k Optional Number of top elements to look at for computing precision,
 *     default to 1.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
Future<U> inTopKAsync<T extends Tensor, U extends Tensor>(
  T predictions,
  U targets, {
  int k = 1,
}) async {
  final $predictions = convertToTensor(predictions, 'predictions', 'inTopK');
  final $targets = convertToTensor(targets, 'targets', 'inTopK');

  assert_(
      $predictions.rank > 1,
      () =>
          'inTopK() expects the predictions to be of rank 2 or higher, ' +
          "but got ${$predictions.rank}");
  assert_(
      $predictions.rank - 1 == $targets.rank,
      () =>
          "predictions rank should be 1 larger than " +
          "targets rank, but got predictions rank " +
          "${$predictions.rank} and targets rank ${$targets.rank}");
  assertShapesMatch(
      $predictions.shape.sublistRelaxed(0, $predictions.shape.length - 1),
      $targets.shape,
      "predictions's shape should be align with the targets' shape, " +
          'except the last dimension.');
  final lastDim = $predictions.shape[$predictions.shape.length - 1];
  assert(
      k > 0 && k <= lastDim,
      () =>
          "'k' passed to inTopK() must be > 0 && <= the predictions last " +
          "dimension (${lastDim}), but got ${k}");

  final predictionsVals = await $predictions.data();
  final targetsVals = await $targets.data();

  // Reshape predictionsVals into a 2d tensor [batch, lastDim]
  // and look up topK along lastDim.
  final batch = predictionsVals.length ~/ lastDim;
  final size = lastDim;

  final precision = getTypedArrayFromDType('bool', batch);

  for (int b = 0; b < batch; b++) {
    final offset = b * size;
    final vals = predictionsVals.slice(offset, offset + size);
    final List<IndexedValue<int>> valAndInd = [];
    for (int i = 0; i < vals.length; i++) {
      valAndInd.add(IndexedValue(value: vals[i], index: i));
    }
    valAndInd.sort((a, b) => b.value - a.value);

    precision[b] = 0;
    for (int i = 0; i < k; i++) {
      if (valAndInd[i].index == targetsVals[b]) {
        precision[b] = 1;
        break;
      }
    }
  }

  if (predictions != $predictions) {
    $predictions.dispose();
  }
  if (targets != $targets) {
    $targets.dispose();
  }

  // Output precision has the same shape as targets.
  return tensor(precision, $targets.shape, 'bool') as U;
}

class IndexedValue<V> {
  final int index;
  final V value;
  IndexedValue({
    required this.index,
    required this.value,
  });
}
