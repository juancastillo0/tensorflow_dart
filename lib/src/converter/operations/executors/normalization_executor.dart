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

// import {Scalar, Tensor, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
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
    case 'FusedBatchNorm':
    case 'FusedBatchNormV2':
      {
        return [
          tfOps.batchNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('mean', node, tensorMap, context) as Tensor,
              getParamValue('variance', node, tensorMap, context) as Tensor,
              offset:
                  getParamValue('offset', node, tensorMap, context) as Tensor,
              scale: getParamValue('scale', node, tensorMap, context) as Tensor,
              varianceEpsilon:
                  getParamValue('epsilon', node, tensorMap, context) as double)
        ];
      }
    case 'FusedBatchNormV3':
      {
        return [
          tfOps.batchNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('mean', node, tensorMap, context) as Tensor,
              getParamValue('variance', node, tensorMap, context) as Tensor,
              offset:
                  getParamValue('offset', node, tensorMap, context) as Tensor,
              scale: getParamValue('scale', node, tensorMap, context) as Tensor,
              varianceEpsilon:
                  getParamValue('epsilon', node, tensorMap, context) as double)
        ];
      }
    case 'LRN':
      {
        return [
          tfOps.localResponseNormalization(
              getParamValue('x', node, tensorMap, context)
                  as Tensor, // Tensor3D | Tensor4D,
              depthRadius:
                  getParamValue('radius', node, tensorMap, context) as int,
              bias: getParamValue('bias', node, tensorMap, context) as double,
              alpha: getParamValue('alpha', node, tensorMap, context) as double,
              beta: getParamValue('beta', node, tensorMap, context) as double)
        ];
      }
    case 'Softmax':
      {
        return [
          tfOps.softmax(getParamValue('x', node, tensorMap, context) as Tensor)
        ];
      }
    case 'LogSoftmax':
      {
        return [
          tfOps.logSoftmax(
              getParamValue('x', node, tensorMap, context) as Tensor)
        ];
      }
    case 'SparseToDense':
      {
        return [
          tfOps.sparseToDense(
            getParamValue('sparseIndices', node, tensorMap, context) as Tensor,
            getParamValue('outputShape', node, tensorMap, context) as Tensor,
            getParamValue('sparseValues', node, tensorMap, context)
                as List<int>,
            defaultValue:
                getParamValue('defaultValue', node, tensorMap, context)
                    as Scalar,
          )
        ];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'normalization';
