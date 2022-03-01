/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

// import {Scalar, Tensor, Tensor1D} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import '_prelude.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;

List<Tensor> executeOp(Node node, NamedTensorsMap tensorMap,
     ExecutionContext context) {
      switch (node.op) {
        case 'StringNGrams': {
          final p = tfOps.string.stringNGrams(
              getParamValue('data', node, tensorMap, context) as Tensor1D,
              getParamValue('dataSplits', node, tensorMap, context) as Tensor,
              getParamValue('separator', node, tensorMap, context) as String,
              getParamValue('nGramWidths', node, tensorMap, context) as
                  number[],
              getParamValue('leftPad', node, tensorMap, context) as String,
              getParamValue('rightPad', node, tensorMap, context) as String,
              getParamValue('padWidth', node, tensorMap, context) as number,
              getParamValue(
                  'preserveShortSequences', node, tensorMap, context) as
                  bool);
          return [p.nGrams, p.nGramsSplits];
        }
        case 'StringSplit': {
          final p = tfOps.string.stringSplit(
              getParamValue('input', node, tensorMap, context) as Tensor1D,
              getParamValue('delimiter', node, tensorMap, context) as Scalar,
              getParamValue('skipEmpty', node, tensorMap, context) as bool);
          return [p.indices, p.values, p.shape];
        }
        case 'StringToHashBucketFast': {
          final output = tfOps.string.stringToHashBucketFast(
              getParamValue('input', node, tensorMap, context) as Tensor,
              getParamValue('numBuckets', node, tensorMap, context) as int);
          return [output];
        }
        default:
          throw StateError('Node type ${node.op} is not implemented');
      }
    }

const CATEGORY = 'string';
