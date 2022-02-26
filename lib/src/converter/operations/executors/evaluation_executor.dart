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

// import {Tensor} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

List<Tensor> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  switch (node.op) {
    case 'TopKV2':
      {
        final x = getParamValue('x', node, tensorMap, context) as Tensor;
        final k = getParamValue('k', node, tensorMap, context) as int;
        final sorted =
            getParamValue('sorted', node, tensorMap, context) as bool;
        final result = tfOps.topk(x, k: k, sorted: sorted);
        return [result.values, result.indices];
      }
    case 'Unique':
      {
        final x = getParamValue('x', node, tensorMap, context) as Tensor;
        final result = tfOps.unique(x);
        return [result.values, result.indices];
      }
    case 'UniqueV2':
      {
        final x = getParamValue('x', node, tensorMap, context) as Tensor;
        final axis = getParamValue('axis', node, tensorMap, context) as int;
        final result = tfOps.unique(x, axis);
        return [result.values, result.indices];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'evaluation';
