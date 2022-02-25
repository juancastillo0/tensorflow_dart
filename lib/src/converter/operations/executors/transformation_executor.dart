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

// import {Tensor, Tensor4D} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

List<Tensor> executeOp(
    Node node, NamedTensorsMap tensorMap, ExecutionContext context) {
  switch (node.op) {
    case 'Cast':
      {
        return [
          tfOps.cast(
            getParamValue('x', node, tensorMap, context) as Tensor,
            getParamValue('dtype', node, tensorMap, context)
                as DataType, //'int32' |'float32' | 'bool'
          )
        ];
      }
    case 'ExpandDims':
      {
        final axis = getParamValue('axis', node, tensorMap, context) as int;
        return [
          tfOps.expandDims(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)
        ];
      }
    case 'Squeeze':
      {
        final axis =
            getParamValue('axis', node, tensorMap, context) as List<int>;
        return [
          tfOps.squeeze(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)
        ];
      }

    case 'Reshape':
      {
        return [
          tfOps.reshape(getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('shape', node, tensorMap, context) as List<int>)
        ];
      }
    case 'MirrorPad':
      {
        return [
          tfOps.mirrorPad(
            getParamValue('x', node, tensorMap, context) as Tensor,
            getParamValue('padding', node, tensorMap, context)
                as List<List<int>>, // Array<[number, number]>,
            mode: getParamValue('mode', node, tensorMap, context)
                as String, //'reflect' | 'symmetric'
          )
        ];
      }
    case 'PadV2':
    case 'Pad':
      {
        return [
          tfOps.pad(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('padding', node, tensorMap, context)
                  as List<List<int>>, //Array<[number, number]>,
              constantValue:
                  getParamValue('constantValue', node, tensorMap, context)
                      as double)
        ];
      }
    case 'SpaceToBatchND':
      {
        final blockShape =
            getParamValue('blockShape', node, tensorMap, context) as List<int>;
        final paddings = getParamValue('paddings', node, tensorMap, context)
            as List<List<int>>;
        return [
          tfOps.spaceToBatchND(
              getParamValue('x', node, tensorMap, context) as Tensor,
              blockShape,
              paddings)
        ];
      }
    case 'BatchToSpaceND':
      {
        final blockShape =
            getParamValue('blockShape', node, tensorMap, context) as List<int>;
        final crops =
            getParamValue('crops', node, tensorMap, context) as List<List<int>>;
        return [
          tfOps.batchToSpaceND(
              getParamValue('x', node, tensorMap, context) as Tensor,
              blockShape,
              crops)
        ];
      }
    case 'DepthToSpace':
      {
        final blockSize =
            getParamValue('blockSize', node, tensorMap, context) as int;
        final dataFormat =
            (getParamValue('dataFormat', node, tensorMap, context) as String)
                .toUpperCase() as String; // 'NHWC' | 'NCHW';
        return [
          tfOps.depthToSpace(
              getParamValue('x', node, tensorMap, context) as Tensor4D,
              blockSize,
              dataFormat: dataFormat)
        ];
      }
    case 'BroadcastTo':
      {
        return [
          tfOps.broadcastTo(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('shape', node, tensorMap, context) as List<int>)
        ];
      }
    case 'BroadcastArgs':
      {
        return [
          tfOps.broadcastArgs(
              getParamValue('s0', node, tensorMap, context) as Tensor,
              getParamValue('s1', node, tensorMap, context) as Tensor)
        ];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'transformation';
