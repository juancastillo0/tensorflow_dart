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

// import {Tensor, Tensor2D} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import 'package:tensorflow_wasm/src/ops/fused_types.dart' show Activation;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

List<Tensor> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  switch (node.op) {
    case 'BatchMatMul':
    case 'BatchMatMulV2':
    case 'MatMul':
      return [
        tfOps.matMul(getParamValue('a', node, tensorMap, context) as Tensor2D,
            getParamValue('b', node, tensorMap, context) as Tensor2D,
            transposeA:
                getParamValue('transposeA', node, tensorMap, context) as bool,
            transposeB:
                getParamValue('transposeB', node, tensorMap, context) as bool)
      ];

    case 'Einsum':
      return [
        tfOps.einsum(
          getParamValue('equation', node, tensorMap, context) as String,
          [
            ...getParamValue('tensors', node, tensorMap, context)
                as List<Tensor>
          ],
        )
      ];

    case 'Transpose':
      return [
        tfOps.transpose(getParamValue('x', node, tensorMap, context) as Tensor,
            perm: getParamValue('perm', node, tensorMap, context) as List<int>)
      ];

    case '_FusedMatMul':
      final _ops =
          getParamValue('fusedOps', node, tensorMap, context) as List<String>;
      final extraOp = _ops.first;
      final activationFunc = _ops.last;

      final isBiasAdd = extraOp == 'biasadd';
      final isPrelu = activationFunc == 'prelu';

      final numArgs =
          (getParamValue('numArgs', node, tensorMap, context) as int);
      final leakyreluAlpha =
          getParamValue('leakyreluAlpha', node, tensorMap, context) as num?;

      if (isBiasAdd) {
        if (isPrelu && numArgs != 2) {
          throw Exception('Fused MatMul with BiasAdd and Prelu must have two ' +
              'extra arguments: bias and alpha.');
        }
        if (!isPrelu && numArgs != 1) {
          throw Exception(
              'Fused MatMul with BiasAdd must have one extra argument: bias.');
        }
      }
      final _args =
          getParamValue('args', node, tensorMap, context) as List<Tensor>?;
      final biasArg = numArgs >= 1 ? _args!.first : null;
      final preluArg = numArgs >= 2 ? _args!.last : null;

      return [
        tfOps.fused.matMul(
          a: getParamValue('a', node, tensorMap, context) as Tensor2D,
          b: getParamValue('b', node, tensorMap, context) as Tensor2D,
          transposeA:
              getParamValue('transposeA', node, tensorMap, context) as bool,
          transposeB:
              getParamValue('transposeB', node, tensorMap, context) as bool,
          bias: biasArg,
          activation: Activation.values.byName(activationFunc),
          preluActivationWeights: preluArg,
          leakyreluAlpha: leakyreluAlpha?.toDouble(),
        )
      ];

    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'matrices';
