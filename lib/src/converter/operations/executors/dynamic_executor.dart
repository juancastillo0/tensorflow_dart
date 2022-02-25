/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// import {Tensor, Tensor1D, Tensor2D} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpAsyncExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

class _Params {
  final Tensor boxes;
  final Tensor scores;
  final int maxOutputSize;
  final double? iouThreshold;
  final double? scoreThreshold;
  final double? softNmsSigma;

  _Params({
    required this.boxes,
    required this.scores,
    required this.maxOutputSize,
    required this.iouThreshold,
    required this.scoreThreshold,
    required this.softNmsSigma,
  });
}

_Params _nmsParams(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  final boxes = getParamValue('boxes', node, tensorMap, context) as Tensor;
  final scores = getParamValue('scores', node, tensorMap, context) as Tensor;
  final maxOutputSize =
      getParamValue('maxOutputSize', node, tensorMap, context) as int;
  final iouThreshold =
      getParamValue('iouThreshold', node, tensorMap, context) as double?;
  final scoreThreshold =
      getParamValue('scoreThreshold', node, tensorMap, context) as double?;
  final softNmsSigma =
      getParamValue('softNmsSigma', node, tensorMap, context) as double?;

  return _Params(
    boxes: boxes,
    scores: scores,
    maxOutputSize: maxOutputSize,
    iouThreshold: iouThreshold,
    scoreThreshold: scoreThreshold,
    softNmsSigma: softNmsSigma,
  );
}

Future<List<Tensor>> executeOp(
    Node node, NamedTensorsMap tensorMap, ExecutionContext context) async {
  switch (node.op) {
    case 'NonMaxSuppressionV5':
      {
        final p = _nmsParams(node, tensorMap, context);

        final result = await tfOps.image.nonMaxSuppressionWithScoreAsync(
            p.boxes as Tensor2D, p.scores as Tensor1D, p.maxOutputSize,
            iouThreshold: p.iouThreshold,
            scoreThreshold: p.scoreThreshold,
            softNmsSigma: p.softNmsSigma);

        return [result.selectedIndices, result.selectedScores];
      }
    case 'NonMaxSuppressionV4':
      {
        final p = _nmsParams(node, tensorMap, context);

        final padToMaxOutputSize =
            getParamValue('padToMaxOutputSize', node, tensorMap, context)
                as bool;

        final result = await tfOps.image.nonMaxSuppressionPaddedAsync(
            p.boxes as Tensor2D, p.scores as Tensor1D, p.maxOutputSize,
            iouThreshold: p.iouThreshold,
            scoreThreshold: p.scoreThreshold,
            padToMaxOutputSize: padToMaxOutputSize);

        return [result.selectedIndices, result.validOutputs];
      }
    case 'NonMaxSuppressionV3':
    case 'NonMaxSuppressionV2':
      {
        final p = _nmsParams(node, tensorMap, context);

        return [
          await tfOps.image.nonMaxSuppressionAsync(
              p.boxes as Tensor2D, p.scores as Tensor1D, p.maxOutputSize,
              iouThreshold: p.iouThreshold, scoreThreshold: p.scoreThreshold)
        ];
      }
    case 'Where':
      {
        final condition = tfOps.cast(
            (getParamValue('condition', node, tensorMap, context) as Tensor),
            'bool');
        final result = [await tfOps.whereAsync(condition)];
        condition.dispose();
        return result;
      }
    case 'ListDiff':
      {
        return tfOps.setdiff1dAsync(
            getParamValue('x', node, tensorMap, context) as Tensor,
            getParamValue('y', node, tensorMap, context) as Tensor);
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'dynamic';
