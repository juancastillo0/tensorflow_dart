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

// import {cloneTensor, getParamValue, getTensor} from './utils';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

List<Tensor> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  switch (node.op) {
    case 'Const':
      {
        return tensorMap[node.name]!;
      }
    case 'PlaceholderWithDefault':
      final def = getParamValue('default', node, tensorMap, context) as Tensor;
      return [getTensor(node.name, tensorMap, context) ?? def];
    case 'Placeholder':
      return [getTensor(node.name, tensorMap, context)!];
    case 'Identity':
    case 'StopGradient':
    case 'FakeQuantWithMinMaxVars':
      {
        // This op is currently ignored.
        final data = getParamValue('x', node, tensorMap, context) as Tensor;
        return [cloneTensor(data)];
      }
    case 'IdentityN':
      return (getParamValue('x', node, tensorMap, context) as List<Tensor>)
          .map((Tensor t) => cloneTensor(t))
          .toList();
    case 'Snapshot':
      final snapshot = (getParamValue('x', node, tensorMap, context) as Tensor);
      return [cloneTensor(snapshot)];
    case 'Shape':
      return [
        tfOps.tensor1d(
            (getParamValue('x', node, tensorMap, context) as Tensor).shape,
            'int32')
      ];
    case 'ShapeN':
      return (getParamValue('x', node, tensorMap, context) as List<Tensor>)
          .map((Tensor t) => tfOps.tensor1d(t.shape))
          .toList();
    case 'Size':
      return [
        tfOps.scalar(
            (getParamValue('x', node, tensorMap, context) as Tensor).size,
            'int32')
      ];
    case 'Rank':
      return [
        tfOps.scalar(
            (getParamValue('x', node, tensorMap, context) as Tensor).rank,
            'int32')
      ];
    case 'NoOp':
      return [tfOps.scalar(1)];
    case 'Print':
      final input = getParamValue('x', node, tensorMap, context) as Tensor;
      final data =
          getParamValue('data', node, tensorMap, context) as List<Tensor>;
      final message =
          getParamValue('message', node, tensorMap, context) as String;
      final summarize =
          getParamValue('summarize', node, tensorMap, context) as number;
      util.log.warning('The graph has a tf.print() operation,' +
          'usually used for debugging, which slows down performance.');
      util.log.info(message);
      for (int i = 0; i < data.length; i++) {
        util.log.info(
            Array.prototype.slice.call(data[i].dataSync()).slice(0, summarize));
      }
      return [input];

    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'graph';
