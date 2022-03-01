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

import 'package:tensorflow_wasm/src/converter/executor/execution_context.dart';
import 'package:tensorflow_wasm/src/converter/executor/resource_manager.dart';
import 'package:tensorflow_wasm/src/converter/operations/types.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' show clone;
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:collection/collection.dart';

// import {clone, Tensor, util} from '@tensorflow/tfjs-core';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {ResourceManager} from '../../executor/resource_manager';
// import {Node, ValueType} from '../types';

ValueType? getParamValue(
  String paramName,
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context, [
  ResourceManager? resourceManager,
]) {
  final inputParam = node.inputParams[paramName];
  if (inputParam != null && inputParam.inputIndexStart != null) {
    final start = inputParam.inputIndexStart!;
    final end = inputParam.inputIndexEnd == 0
        ? null
        : (inputParam.inputIndexEnd == null
            ? start + 1
            : inputParam.inputIndexEnd);
    if (inputParam.type == 'tensor') {
      return getTensor(
        node.inputNames[start],
        tensorMap,
        context,
        resourceManager,
      );
    }
    if (inputParam.type == 'tensors') {
      final inputs = node.inputNames.slice(start, end);

      return inputs
          .map((name) => getTensor(name, tensorMap, context, resourceManager));
    }
    final tensor = getTensor(
      node.inputNames.slice(start)[0],
      tensorMap,
      context,
      resourceManager,
    )!;
    final data = tensor.dataSync();
    return inputParam.type == 'number'
        ? data[0]
        : util.toNestedArray(tensor.shape, data);
  }
  final attrParam = node.attrParams[paramName];
  return attrParam?.value;
}

/**
 * Retrieve the tensor from tensorsMap based on input name.
 * @param name Node input name
 * @param tensorsMap Tensors map keyed by the node
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
Tensor? getTensor(
  String name,
  NamedTensorsMap tensorsMap,
  ExecutionContext context, [
  ResourceManager? resourceManager,
]) {
  final n = parseNodeName(name);
  final nodeName = n.nodeName;
  final index = n.index;

  if (resourceManager != null) {
    final tensor = resourceManager.getHashTableHandleByName(nodeName);
    if (tensor != null) {
      return tensor;
    }
  }

  for (final contextId in context.currentContextIds) {
    final list = tensorsMap[getNodeNameWithContextId(nodeName, contextId)];
    if (list != null) return list[index];
  }
  return null;
}

/**
 * Retrieve the tensors based on input name for current context.
 * @param name Node input name
 * @param tensorsMap Tensors map keyed by the node
 */
List<Tensor>? getTensorsForCurrentContenxt(
  String name,
  NamedTensorsMap tensorsMap,
  ExecutionContext context,
) {
  return tensorsMap[getNodeNameWithContextId(name, context.currentContextId)];
}

/**
 * Returns the node name, outputName and index from the Node input name.
 * @param inputName The input name of the node, in format of
 * node_name:output_index, i.e. MatMul:0, if the output_index is not set, it is
 * default to 0.
 * If the input name contains output name i.e. StringSplit:indices:0, it will
 * return ['StringSplit', 0, 'indices'].
 */
NodeName getNodeNameAndIndex(
  String inputName, [
  ExecutionContext? context,
]) {
  final nodeName = parseNodeName(inputName);

  return NodeName(
    nodeName: getNodeNameWithContextId(
      nodeName.nodeName,
      context?.currentContextId,
    ),
    index: nodeName.index,
    outputName: nodeName.outputName,
  );
}

String getNodeNameWithContextId(String name, [String? contextId]) {
  return contextId != null ? '${name}-${contextId}' : name;
}

class NodeName {
  final String nodeName;
  final int index;
  final String? outputName;

  NodeName({
    required this.nodeName,
    required this.index,
    required this.outputName,
  });
}

NodeName parseNodeName(String name) {
  final parts = name.split(':');
  if (parts.length == 1) {
    return NodeName(
      nodeName: name,
      index: 0,
      outputName: null,
    );
  }

  final nodeName = parts[0];
  final outputName = parts.length == 3 ? parts[1] : null;
  final index = int.parse(parts[parts.length - 1]);
  return NodeName(
    nodeName: nodeName,
    index: index,
    outputName: outputName,
  );
}

List<List<int>> split(List<int> arr, int size) {
  final List<List<int>> res = [];
  for (int i = 0; i < arr.length; i += size) {
    res.add(arr.slice(i, i + size));
  }
  return res;
}

ValueType? getPadding(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  var pad = getParamValue('pad', node, tensorMap, context);
  if (pad == 'explicit') {
    // This is 1d array, we need to convert it to 2d array
    pad = getParamValue('explicitPaddings', node, tensorMap, context);
    final explicitPadding = [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0]
    ];
    for (int i = 0; i < 4; i++) {
      explicitPadding[i][0] = (pad as List<int>)[i * 2];
      explicitPadding[i][1] = (pad as List<int>)[i * 2 + 1];
    }
    return explicitPadding;
  }
  return pad;
}

/**
 *  Reuse the tensor if it is marked as keep, otherwise clone the tensor to
 *  avoid disposal. This is important for TensorArray and TensorList ops, since
 *  internally they use a tensor as the id for TensorArray and TensorList, and
 * to simplify lookup, they also use Tensor.id as the key to the internal map.
 * These id tensors have been marked as kept in the backend, we need avoid clone
 * them in order to create new Tensor.id.
 * @param tensor
 */
Tensor cloneTensor(Tensor tensor) {
  return tensor.kept ? tensor : clone(tensor);
}
