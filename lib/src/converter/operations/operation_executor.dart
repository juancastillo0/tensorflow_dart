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

// import * as tfc from '@tensorflow/tfjs-core';

// import {NamedTensorsMap} from '../data/types';
// import {ExecutionContext} from '../executor/execution_context';
// import {ResourceManager} from '../executor/resource_manager';

// import {NodeValueImpl} from './custom_op/node_value_impl';
import 'custom_op/node_value_impl.dart';
import 'custom_op/register.dart' show getRegisteredOp;
import './executors/arithmetic_executor.dart' as arithmetic;
import './executors/basic_math_executor.dart' as basicMath;
import './executors/control_executor.dart' as control;
import './executors/convolution_executor.dart' as convolution;
import './executors/creation_executor.dart' as creation;
import './executors/dynamic_executor.dart' as dynamic_;
import './executors/evaluation_executor.dart' as evaluation;
import './executors/graph_executor.dart' as graph;
import './executors/hash_table_executor.dart' as hashTable;
import './executors/image_executor.dart' as image;
import './executors/logical_executor.dart' as logical;
import './executors/matrices_executor.dart' as matrices;
import './executors/normalization_executor.dart' as normalization;
import './executors/reduction_executor.dart' as reduction;
import './executors/slice_join_executor.dart' as sliceJoin;
import './executors/sparse_executor.dart' as sparse;
// import * as spectral from './executors/spectral_executor';
import './executors/string_executor.dart' as string;
import './executors/transformation_executor.dart' as transformation;
// import {Node} from './types';

import 'dart:async';

import 'package:tensorflow_wasm/src/converter/executor/execution_context.dart';
import 'package:tensorflow_wasm/src/converter/executor/resource_manager.dart';
import 'package:tensorflow_wasm/src/converter/operations/types.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfc;

/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
FutureOr<List<tfc.Tensor>> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context, [
  ResourceManager? resourceManager,
]) {
  FutureOr<List<Tensor>> _d(
    Node node,
    NamedTensorsMap tensorMap,
    ExecutionContext context,
  ) {
    switch (node.category) {
      case 'arithmetic':
        return tfc.tidy(() => arithmetic.executeOp(node, tensorMap, context));
      case 'basic_math':
        return tfc.tidy(() => basicMath.executeOp(node, tensorMap, context));
      case 'control':
        return control.executeOp(node, tensorMap, context);
      case 'convolution':
        return tfc.tidy(() => convolution.executeOp(node, tensorMap, context));
      case 'creation':
        return tfc.tidy(() => creation.executeOp(node, tensorMap, context));
      case 'dynamic':
        return dynamic_.executeOp(node, tensorMap, context);
      case 'evaluation':
        return tfc.tidy(() => evaluation.executeOp(node, tensorMap, context));
      case 'image':
        return tfc.tidy(() => image.executeOp(node, tensorMap, context));
      case 'graph':
        return tfc.tidy(() => graph.executeOp(node, tensorMap, context));
      case 'logical':
        return tfc.tidy(() => logical.executeOp(node, tensorMap, context));
      case 'matrices':
        return tfc.tidy(() => matrices.executeOp(node, tensorMap, context));
      case 'normalization':
        return tfc
            .tidy(() => normalization.executeOp(node, tensorMap, context));
      case 'reduction':
        return tfc.tidy(() => reduction.executeOp(node, tensorMap, context));
      case 'slice_join':
        return tfc.tidy(() => sliceJoin.executeOp(node, tensorMap, context));
      case 'sparse':
        return tfc.tidy(() => sparse.executeOp(node, tensorMap, context));
      case 'spectral':
        // TODO: spectral
        // return tfc.tidy(() => spectral.executeOp(node, tensorMap, context));
        throw UnimplementedError();
      case 'string':
        return tfc.tidy(() => string.executeOp(node, tensorMap, context));
      case 'transformation':
        return tfc
            .tidy(() => transformation.executeOp(node, tensorMap, context));
      case 'hash_table':
        return hashTable.executeOp(node, tensorMap, context, resourceManager!);
      case 'custom':
        final opMapper = getRegisteredOp(node.op);
        if (opMapper != null && opMapper.customExecutor != null) {
          return _then<Tensors, List<Tensor>>(
            opMapper.customExecutor!(NodeValueImpl(node, tensorMap, context)),
            (t) => t.toTensorList(),
          );
        } else {
          throw StateError("Custom op ${node.op} is not registered.");
        }
      default:
        throw StateError("Unknown op '${node.op}'. File an issue at " +
            "https://github.com/tensorflow/tfjs/issues so we can add it" +
            ", or register a custom execution with tf.registerOp()");
    }
  }

  final value = _d(node, tensorMap, context);
  return value;
}

FutureOr<O> _then<T, O>(FutureOr<T> fut, FutureOr<O> Function(T) map) {
  if (fut is Future<T>) return fut.then(map);
  return map(fut);
}
