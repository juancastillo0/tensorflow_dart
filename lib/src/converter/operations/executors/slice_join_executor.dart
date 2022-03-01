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

// import {Scalar, Tensor, Tensor1D, tidy, util} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

List<Tensor> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  switch (node.op) {
    case 'ConcatV2':
    case 'Concat':
      {
        final n = getParamValue('n', node, tensorMap, context) as int;
        final axis = getParamValue('axis', node, tensorMap, context) as int;
        var inputs =
            getParamValue('tensors', node, tensorMap, context) as List<Tensor>;
        inputs = inputs.sublistRelaxed(0, n);
        return [tfOps.concat(inputs, axis)];
      }
    case 'Gather':
      {
        final input = getParamValue('x', node, tensorMap, context) as Tensor;
        final indices = getParamValue('indices', node, tensorMap, context)
            as Tensor; // Tensor1D
        return [tfOps.gather(input, tfOps.cast(indices, 'int32'), axis: 0)];
      }
    case 'GatherV2':
      {
        final axis =
            getParamValue('axis', node, tensorMap, context) as int? ?? 0;
        final batchDims =
            getParamValue('batchDims', node, tensorMap, context) as int? ?? 0;
        final input = getParamValue('x', node, tensorMap, context) as Tensor;
        final indices = getParamValue('indices', node, tensorMap, context)
            as Tensor; // Tensor1D
        return [
          tfOps.gather(input, tfOps.cast(indices, 'int32'),
              axis: axis, batchDims: batchDims)
        ];
      }
    case 'Reverse':
      {
        final dims =
            getParamValue('dims', node, tensorMap, context) as List<bool>;
        final List<int> axis = [];
        for (int i = 0; i < dims.length; i++) {
          if (dims[i]) {
            axis.add(i);
          }
        }
        final input = getParamValue('x', node, tensorMap, context) as Tensor;
        return [tfOps.reverse(input, axis)];
      }
    case 'ReverseV2':
      {
        final axis =
            getParamValue('axis', node, tensorMap, context) as List<int>;
        final input = getParamValue('x', node, tensorMap, context) as Tensor;
        return [tfOps.reverse(input, axis)];
      }
    case 'Slice':
      {
        // tslint:disable-next-line:no-any
        final begin =
            getParamValue('begin', node, tensorMap, context) as List<int>;
        // tslint:disable-next-line:no-any
        final size =
            getParamValue('size', node, tensorMap, context) as List<int>?;
        return [
          tfOps.slice(getParamValue('x', node, tensorMap, context) as Tensor,
              begin, size)
        ];
      }
    case 'StridedSlice':
      {
        final begin =
            getParamValue('begin', node, tensorMap, context) as List<int>;
        final end = getParamValue('end', node, tensorMap, context) as List<int>;
        final strides =
            getParamValue('strides', node, tensorMap, context) as List<int>?;
        final beginMask =
            getParamValue('beginMask', node, tensorMap, context) as int? ?? 0;
        final endMask =
            getParamValue('endMask', node, tensorMap, context) as int? ?? 0;
        final ellipsisMask =
            getParamValue('ellipsisMask', node, tensorMap, context) as int? ??
                0;
        final newAxisMask =
            getParamValue('newAxisMask', node, tensorMap, context) as int? ?? 0;
        final shrinkAxisMask =
            getParamValue('shrinkAxisMask', node, tensorMap, context) as int? ??
                0;
        final tensor = getParamValue('x', node, tensorMap, context) as Tensor;

        return [
          tfOps.stridedSlice(
            tensor,
            begin,
            end,
            strides: strides,
            beginMask: beginMask,
            endMask: endMask,
            ellipsisMask: ellipsisMask,
            newAxisMask: newAxisMask,
            shrinkAxisMask: shrinkAxisMask,
          )
        ];
      }
    case 'Pack':
      {
        return tfOps.tidy(() {
          final axis = getParamValue('axis', node, tensorMap, context) as int;
          final tensors = getParamValue('tensors', node, tensorMap, context)
              as List<Tensor>;
          // Reshape the tensors to the first tensor's shape if they don't
          // match.
          final shape = tensors[0].shape;
          final squeezedShape = tfOps.squeeze(tensors[0]).shape;
          final mapped = tensors.map((tensor) {
            final sameShape = util.arraysEqual(tensor.shape, shape);
            if (!sameShape &&
                !util.arraysEqual(tfOps.squeeze(tensor).shape, squeezedShape)) {
              throw Exception('the input tensors shape does not match');
            }
            return sameShape ? tensor : tfOps.reshape(tensor, shape);
          }).toList();
          return [tfOps.stack(mapped, axis)];
        });
      }
    case 'Unpack':
      {
        final axis = getParamValue('axis', node, tensorMap, context) as int;
        final tensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        return tfOps.unstack(tensor, axis);
      }
    case 'Tile':
      {
        final reps =
            getParamValue('reps', node, tensorMap, context) as List<int>;
        return [
          tfOps.tile(
              getParamValue('x', node, tensorMap, context) as Tensor, reps)
        ];
      }
    case 'Split':
    case 'SplitV':
      {
        final axis =
            getParamValue('axis', node, tensorMap, context) as int? ?? 0;
        final numOrSizeSplits =
            getParamValue('numOrSizeSplits', node, tensorMap, context);
        final tensor = getParamValue('x', node, tensorMap, context) as Tensor;

        return tfOps.split(
            tensor,
            numOrSizeSplits is int
                ? [numOrSizeSplits]
                : numOrSizeSplits as List<int>,
            axis: axis);
      }
    case 'ScatterNd':
      {
        final indices =
            getParamValue('indices', node, tensorMap, context) as Tensor;
        final values =
            getParamValue('values', node, tensorMap, context) as Tensor;
        final shape =
            getParamValue('shape', node, tensorMap, context) as List<int>;
        return [tfOps.scatterND(indices, values, shape)];
      }
    case 'GatherNd':
      {
        final x = getParamValue('x', node, tensorMap, context) as Tensor;
        final indices =
            getParamValue('indices', node, tensorMap, context) as Tensor;
        return [tfOps.gatherND(x, indices)];
      }
    case 'SparseToDense':
      {
        final indices =
            getParamValue('sparseIndices', node, tensorMap, context) as Tensor;
        final shape =
            getParamValue('outputShape', node, tensorMap, context) as List<int>;
        final sparseValues =
            getParamValue('sparseValues', node, tensorMap, context) as Tensor;
        final defaultValue =
            getParamValue('defaultValue', node, tensorMap, context)
                as Tensor; // Scalar;
        return [
          tfOps.sparseToDense(indices, sparseValues, shape,
              defaultValue: sparseValues.dtype == defaultValue.dtype
                  ? defaultValue
                  : tfOps.cast(defaultValue, sparseValues.dtype))
        ];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'slice_join';
