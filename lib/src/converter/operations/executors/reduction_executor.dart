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
    case 'Max':
      {
        final axis = getParamValueList<int>('axis', node, tensorMap, context)!;
        final keepDims =
            getParamValue('keepDims', node, tensorMap, context) as bool;
        return [
          tfOps.max(getParamValue('x', node, tensorMap, context) as Tensor,
              axis, keepDims)
        ];
      }
    case 'Mean':
      {
        final axis = getParamValueList<int>('axis', node, tensorMap, context);
        final keepDims =
            getParamValue('keepDims', node, tensorMap, context) as bool;
        return [
          tfOps.mean(getParamValue('x', node, tensorMap, context) as Tensor,
              axis, keepDims)
        ];
      }
    case 'Min':
      {
        final axis = getParamValueList<int>('axis', node, tensorMap, context);
        final keepDims =
            getParamValue('keepDims', node, tensorMap, context) as bool;
        return [
          tfOps.min(getParamValue('x', node, tensorMap, context) as Tensor,
              axis, keepDims)
        ];
      }
    case 'Sum':
      {
        final axis = getParamValueList<int>('axis', node, tensorMap, context);
        final keepDims =
            getParamValue('keepDims', node, tensorMap, context) as bool;
        return [
          tfOps.sum(getParamValue('x', node, tensorMap, context) as Tensor,
              axis, keepDims)
        ];
      }
    case 'All':
      {
        final axis = getParamValueList<int>('axis', node, tensorMap, context);
        final keepDims =
            getParamValue('keepDims', node, tensorMap, context) as bool;
        return [
          tfOps.all(getParamValue('x', node, tensorMap, context) as Tensor,
              axis, keepDims)
        ];
      }
    case 'Any':
      {
        final axis = getParamValueList<int>('axis', node, tensorMap, context);
        final keepDims =
            getParamValue('keepDims', node, tensorMap, context) as bool;
        return [
          tfOps.any(getParamValue('x', node, tensorMap, context) as Tensor,
              axis, keepDims)
        ];
      }
    case 'ArgMax':
      {
        final axis = getParamValue('axis', node, tensorMap, context) as int;
        return [
          tfOps.argMax(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)
        ];
      }
    case 'ArgMin':
      {
        final axis = getParamValue('axis', node, tensorMap, context) as int;
        return [
          tfOps.argMin(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)
        ];
      }
    case 'Prod':
      {
        final axis = getParamValueList<int>('axis', node, tensorMap, context);
        final keepDims =
            getParamValue('keepDims', node, tensorMap, context) as bool;
        return [
          tfOps.prod(getParamValue('x', node, tensorMap, context) as Tensor,
              axis, keepDims)
        ];
      }
    case 'Cumsum':
      {
        final axis = getParamValue('axis', node, tensorMap, context) as int;
        final exclusive =
            getParamValue('exclusive', node, tensorMap, context) as bool;
        final reverse =
            getParamValue('reverse', node, tensorMap, context) as bool;
        return [
          tfOps.cumsum(
            getParamValue('x', node, tensorMap, context) as Tensor,
            axis: axis,
            exclusive: exclusive,
            reverse: reverse,
          )
        ];
      }
    case 'Bincount':
      final x = getParamValue('x', node, tensorMap, context) as Tensor1D;
      final weights =
          getParamValue('weights', node, tensorMap, context) as Tensor1D;
      final size = getParamValue('size', node, tensorMap, context) as int;

      return [tfOps.bincount(x, weights, size)];
    case 'DenseBincount':
      {
        final x = getParamValue('x', node, tensorMap, context)
            as Tensor; // TODO: Tensor1D | Tensor2D;
        final weights = getParamValue('weights', node, tensorMap, context)
            as Tensor; // TODO: Tensor1D | Tensor2D;
        final size = getParamValue('size', node, tensorMap, context) as int;

        final binaryOutput =
            getParamValue('binaryOutput', node, tensorMap, context) as bool;

        return [
          tfOps.denseBincount(x, weights, size, binaryOutput: binaryOutput)
        ];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'reduction';
