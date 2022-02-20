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

// import {DataType, Tensor, Tensor1D} from '@tensorflow/tfjs-core';
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
    case 'Fill':
      {
        final shape =
            getParamValue('shape', node, tensorMap, context) as List<int>;
        final dtype =
            getParamValue('dtype', node, tensorMap, context) as DataType;
        final value =
            getParamValue('value', node, tensorMap, context) as number;
        return [tfOps.fill(shape, value, dtype)];
      }
    case 'LinSpace':
      {
        final start =
            getParamValue('start', node, tensorMap, context) as number;
        final stop = getParamValue('stop', node, tensorMap, context) as number;
        final num = getParamValue('num', node, tensorMap, context) as number;
        return [tfOps.linspace(start, stop, num)];
      }
    case 'Multinomial':
      {
        final logits =
            getParamValue('logits', node, tensorMap, context) as Tensor1D;
        final numSamples =
            getParamValue('numSamples', node, tensorMap, context) as number;
        final seed = getParamValue('seed', node, tensorMap, context) as number;
        return [tfOps.multinomial(logits, numSamples, seed)];
      }
    case 'OneHot':
      {
        final indices =
            getParamValue('indices', node, tensorMap, context) as Tensor1D;
        final depth =
            getParamValue('depth', node, tensorMap, context) as number;
        final onValue =
            getParamValue('onValue', node, tensorMap, context) as number;
        final offValue =
            getParamValue('offValue', node, tensorMap, context) as number;
        return [tfOps.oneHot(indices, depth, onValue, offValue)];
      }
    case 'Ones':
      {
        return [
          tfOps.ones(
              getParamValue('shape', node, tensorMap, context) as List<int>,
              getParamValue('dtype', node, tensorMap, context) as DataType)
        ];
      }
    case 'OnesLike':
      {
        return [
          tfOps.onesLike(getParamValue('x', node, tensorMap, context) as Tensor)
        ];
      }
    case 'RandomUniform':
      {
        return [
          tfOps.randomUniform(
              // tslint:disable-next-line:no-any
              getParamValue('shape', node, tensorMap, context) as any,
              getParamValue('minval', node, tensorMap, context) as number,
              getParamValue('maxval', node, tensorMap, context) as number,
              getParamValue('dtype', node, tensorMap, context) as DataType)
        ];
      }
    case 'Range':
      {
        final start =
            getParamValue('start', node, tensorMap, context) as number;
        final stop = getParamValue('stop', node, tensorMap, context) as number;
        final step = getParamValue('step', node, tensorMap, context) as number;
        return [
          tfOps.range(
            start, stop, step,
            getParamValue('dtype', node, tensorMap, context)
                as String, // 'float32' | 'int32',
          )
        ];
      }
    case 'TruncatedNormal':
      {
        final shape =
            getParamValue('shape', node, tensorMap, context) as List<int>;
        final mean = getParamValue('mean', node, tensorMap, context) as number;
        final stdDev =
            getParamValue('stdDev', node, tensorMap, context) as number;
        final seed = getParamValue('seed', node, tensorMap, context) as number;
        return [
          tfOps.truncatedNormal(
              shape,
              mean,
              stdDev,
              getParamValue('dtype', node, tensorMap, context)
                  as String, // 'float32' | 'int32',
              seed)
        ];
      }
    case 'Zeros':
      {
        return [
          tfOps.zeros(
              getParamValue('shape', node, tensorMap, context) as List<int>,
              getParamValue('dtype', node, tensorMap, context) as DataType)
        ];
      }
    case 'ZerosLike':
      {
        return [
          tfOps
              .zerosLike(getParamValue('x', node, tensorMap, context) as Tensor)
        ];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'creation';
