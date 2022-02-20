/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import 'package:tensorflow_wasm/src/converter/operations/executors/utils.dart';
import 'package:tensorflow_wasm/src/converter/operations/types.dart';
import 'package:tensorflow_wasm/src/tensor.dart';

// import {NamedTensorMap} from '@tensorflow/tfjs-core';

// import {NamedTensorsMap} from '../data/types';
// import {parseNodeName} from '../operations/executors/utils';
// import {Graph, Node} from '../operations/types';

class ExecutionInfo {
  final NamedTensorMap inputs;
  final List<Node> outputs;
  final Set<String> usedNodes;
  final List<String> missingInputs;
  final Node? dynamicNode;
  final List<String>? syncInputs;

  const ExecutionInfo({
    required this.inputs,
    required this.outputs,
    required this.usedNodes,
    required this.missingInputs,
    required this.dynamicNode,
    required this.syncInputs,
  });
}

/**
 * Given graph inputs and desired outputs, find the minimal set of nodes
 * to execute in order to compute the outputs. In addition return other useful
 * info such:
 * - Missing inputs needed to compute the output.
 * - Whether the subgraph contains dynamic ops (control flow, dynamic shape).
 * - Alternative inputs in order to avoid async (dynamic op) execution.
 */
ExecutionInfo getExecutionSubgraph(
  NamedTensorMap inputs,
  List<Node> outputs,
  NamedTensorsMap weightMap, [
  List<Node>? initNodes,
]) {
  final Set<String> usedNodes = {};
  final List<String> missingInputs = [];
  Node? dynamicNode = null;
  List<String>? syncInputs = null;

  // Start with the outputs, going backwards and find all the nodes that are
  // needed to compute those outputs.
  final Set<String> seen = {};
  final inputNodeNames =
      inputs.keys.map((name) => parseNodeName(name).nodeName).toList();

  List<String> initNodeNames = [];
  if (initNodes != null) {
    initNodeNames =
        initNodes.map((node) => parseNodeName(node.name).nodeName).toList();
  }

  final frontier = [...outputs];
  while (frontier.length > 0) {
    final node = frontier.removeLast();
    if (isControlFlow(node) || isDynamicShape(node) || isHashTable(node)) {
      if (dynamicNode == null) {
        dynamicNode = node;
        syncInputs = dynamicNode.children
            .map((child) => child.name)
            .where((name) => usedNodes.contains(name))
            .toList();
      }
    }
    usedNodes.add(node.name);

    // Weights are dead end since we already have their values.
    if (weightMap[node.name] != null) {
      continue;
    }
    // This node is a dead end since it's one of the user-provided inputs.
    if (inputNodeNames.indexOf(node.name) != -1) {
      continue;
    }
    // This node is a dead end since it doesn't have any inputs.
    if (initNodeNames.indexOf(node.name) != -1) {
      continue;
    }
    if (node.inputs.length == 0) {
      missingInputs.add(node.name);
      continue;
    }
    node.inputs.forEach((input) {
      // Don't add to the frontier if it is already there.
      if (seen.contains(input.name)) {
        return;
      }
      seen.add(input.name);
      frontier.add(input);
    });
  }
  return ExecutionInfo(
    inputs: inputs,
    outputs: outputs,
    usedNodes: usedNodes,
    missingInputs: missingInputs,
    dynamicNode: dynamicNode,
    syncInputs: syncInputs,
  );
}

/**
 * Given the execution info, return a list of nodes in topological order that
 * need to be executed to compute the output.
 */
List<Node> getNodesInTopologicalOrder(
  Graph graph,
  NamedTensorsMap weightMap,
  ExecutionInfo executionInfo,
) {
  final usedNodes = executionInfo.usedNodes;
  final inputs = executionInfo.inputs;

  final List<Node> frontier = [];
  final inputNodes = inputs.keys
      .map((name) => parseNodeName(name).nodeName)
      .map((name) => graph.nodes[name]!);
  final initNodes = graph.initNodes;

  inputNodes.forEach((input) {
    if (usedNodes.contains(input.name)) {
      frontier.add(input);
    }
  });
  graph.weights.forEach((weight) {
    if (usedNodes.contains(weight.name)) {
      frontier.add(weight);
    }
  });
  if (initNodes != null) {
    initNodes.forEach((node) {
      if (usedNodes.contains(node.name)) {
        frontier.add(node);
      }
    });
  }
  final Set<String> seen = {};
  final List<Node> orderedNodes = [];
  while (frontier.length > 0) {
    final node = frontier.removeLast();
    seen.add(node.name);
    if (!weightMap.containsKey(node.name)) {
      orderedNodes.add(node);
    }
    node.children.forEach((child) {
      if (!seen.contains(child.name) &&
          usedNodes.contains(child.name) &&
          child.inputs.every((input) => seen.contains(input.name))) {
        frontier.add(child);
      }
    });
  }
  return orderedNodes;
}

const CONTROL_FLOW_OPS = [
  'Switch',
  'Merge',
  'Enter',
  'Exit',
  'NextIteration',
  'StatelessIf',
  'StatelessWhile',
  'if',
  'While'
];
const DYNAMIC_SHAPE_OPS = [
  'NonMaxSuppressionV2',
  'NonMaxSuppressionV3',
  'NonMaxSuppressionV5',
  'Where'
];
const HASH_TABLE_OPS = [
  'HashTable',
  'HashTableV2',
  'LookupTableImport',
  'LookupTableImportV2',
  'LookupTableFind',
  'LookupTableFindV2',
  'LookupTableSize',
  'LookupTableSizeV2'
];

bool isControlFlow(Node node) {
  return CONTROL_FLOW_OPS.indexOf(node.op) >= 0;
}

bool isDynamicShape(Node node) {
  return DYNAMIC_SHAPE_OPS.indexOf(node.op) >= 0;
}

bool isHashTable(Node node) {
  return HASH_TABLE_OPS.indexOf(node.op) >= 0;
}
