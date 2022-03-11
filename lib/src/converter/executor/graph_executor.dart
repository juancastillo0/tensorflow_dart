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

// import {DataType, env, NamedTensorMap, Tensor, tidy, util} from '@tensorflow/tfjs-core';

// import {ISignatureDef} from '../data/compiled_api';
// import {NamedTensorsMap, TensorArrayMap, TensorInfo, TensorListMap} from '../data/types';
// import {getNodeNameAndIndex, getParamValue, getTensor, getTensorsForCurrentContenxt, parseNodeName} from '../operations/executors/utils';
// import {executeOp} from '../operations/operation_executor';
// import {Graph, Node} from '../operations/types';

// import {ExecutionContext, ExecutionContextInfo} from './execution_context';
// import {getExecutionSubgraph, getNodesInTopologicalOrder, isControlFlow} from './model_analysis';
// import {ResourceManager} from './resource_manager';
// import {FunctionExecutor} from './types';

import 'package:tensorflow_wasm/src/converter/data/compiled_api.dart'
    show ISignatureDef;
import 'package:tensorflow_wasm/src/converter/executor/execution_context.dart';
import 'package:tensorflow_wasm/src/converter/executor/model_analysis.dart';
import 'package:tensorflow_wasm/src/converter/executor/resource_manager.dart';
import 'package:tensorflow_wasm/src/converter/operations/executors/utils.dart';
import 'package:tensorflow_wasm/src/converter/operations/operation_executor.dart';
import 'package:tensorflow_wasm/src/converter/operations/types.dart';
import 'package:tensorflow_wasm/src/model_types.dart';
import 'package:tensorflow_wasm/src/tensor.dart' hide TensorInfo;
import 'package:tensorflow_wasm/tensorflow_wasm.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

class NodeWithContexts {
  final List<ExecutionContextInfo> contexts;
  final Node node;

  NodeWithContexts({
    required this.contexts,
    required this.node,
  });
}

class GraphExecutor implements FunctionExecutor {
  final Map<String, List<Node>> _compiledMap = {};
  NamedTensorsMap _weightMap = {};
  List<int> _weightIds = [];
  final ISignatureDef? _signature;
  final List<Node> _inputs;
  final List<Node> _outputs;
  final List<Node>? _initNodes; // Internal init nodes to start initialization.
  final SEPERATOR = ',';
  final Map<String, Graph>? _functions;
  final Map<String, FunctionExecutor> _functionExecutorMap = {};
  ResourceManager? _resourceManager;
  final NamedTensorsMap _intermediateTensors = {};
  Set<int> _keepIds = {};
  NamedTensorsMap? _tensorsMap;
  bool _keepTensorForDebug = false;

  List<int> get weightIds {
    return this.parent?.weightIds ?? this._weightIds;
  }

  Map<String, FunctionExecutor> get functionExecutorMap {
    return this.parent?.functionExecutorMap ?? this._functionExecutorMap;
  }

  NamedTensorsMap get weightMap {
    return this.parent?.weightMap ?? this._weightMap;
  }

  set weightMap(NamedTensorsMap weightMap) {
    final weightIds = weightMap.values
        .expand((tensors) => tensors.map((tensor) => tensor?.id))
        .whereType<int>()
        .toList();
    this._weightIds = weightIds;
    this._weightMap = weightMap;
  }

  /**
   * Set `ResourceManager` shared by executors of a model.
   * @param resourceManager: `ResourceManager` of the `GraphModel`.
   */
  set resourceManager(ResourceManager resourceManager) {
    this._resourceManager = resourceManager;
  }

  List<ModelTensorInfo> get inputs {
    return this._inputs.map((node) {
      return ModelTensorInfo(
        name: node.name,
        shape: node.attrParams['shape']?.value as List<int>?,
        // TODO: should the default be float32?
        dtype: node.attrParams['dtype']?.value as DataType? ?? 'float32',
      );
    }).toList();
  }

  List<ModelTensorInfo> get outputs {
    return this._outputs.map((node) {
      return ModelTensorInfo(
        name: node.name,
        shape: node.attrParams['shape']?.value as List<int>?,
        // TODO: should the default be float32?
        dtype: node.attrParams['dtype']?.value as DataType? ?? 'float32',
      );
    }).toList();
  }

  List<String> get inputNodes {
    return this._inputs.map((node) => node.signatureKey ?? node.name).toList();
  }

  List<String> get outputNodes {
    return this._outputs.map((node) {
      final name = node.signatureKey ?? node.name;
      return node.defaultOutput != null && node.defaultOutput != 0
          ? ("${name}:${node.defaultOutput}")
          : name;
    }).toList();
  }

  Map<String, ISignatureDef>? get functions {
    return this._functions?.keys.fold<Map<String, ISignatureDef>>({},
        (map, key) {
      map[key] = this._functions![key]!.signature!;
      return map;
    });
  }

  final Graph graph;
  final GraphExecutor? parent;
  /**
   *
   * @param graph Graph the model or function graph to be executed.
   * @param parent When building function exector you need to set the parent
   * executor. Since the weights and function executor maps are set at parant
   * level, that function executor can access the function maps and weight maps
   * through the parent.
   */
  GraphExecutor(this.graph, this.parent)
      : this._outputs = graph.outputs,
        this._inputs = graph.inputs,
        this._initNodes = graph.initNodes,
        this._signature = graph.signature,
        this._functions = graph.functions {
    // create sub-graph executors
    if (graph.functions != null) {
      graph.functions!.entries.forEach((e) {
        this._functionExecutorMap[e.key] = GraphExecutor(e.value, this);
      });
    }
  }

  String _getCompilationKey(List<Node> inputs, List<Node> outputs) {
    final sortedInputs = inputs.map((node) => node.name).toList()..sort();
    final sortedOutputs = outputs.map((node) => node.name).toList()..sort();
    return sortedInputs.join(this.SEPERATOR) +
        '--' +
        sortedOutputs.join(this.SEPERATOR);
  }

  /**
   * Compiles the inference graph and returns the minimal set of nodes that are
   * required for execution, in the correct execution order.
   */
  List<Node> _compile(NamedTensorMap inputs, List<Node> outputs) {
    final executionInfo =
        getExecutionSubgraph(inputs, outputs, this.weightMap, this._initNodes);
    final dynamicNode = executionInfo.dynamicNode;
    if (dynamicNode != null) {
      throw Exception(
          "This execution contains the node '${dynamicNode.name}', which has " +
              "the dynamic op '${dynamicNode.op}'. Please use " +
              "model.executeAsync() instead. Alternatively, to avoid the " +
              "dynamic ops, specify the inputs [${executionInfo.syncInputs}]");
    }

    if (executionInfo.missingInputs.length > 0) {
      final outNames = outputs.map((n) => n.name);
      final inNames = inputs.keys;
      throw Exception(
          "Cannot compute the outputs [${outNames}] from the provided inputs " +
              "[${inNames}]. Missing the following inputs: [${executionInfo.missingInputs}]");
    }

    return getNodesInTopologicalOrder(
        this.graph, this.weightMap, executionInfo);
  }

  /**
   * Executes the inference for given input tensors.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs Optional. output node name from the Tensorflow model, if
   * no outputs are specified, the default outputs of the model would be used.
   * You can inspect intermediate nodes of the model by adding them to the
   * outputs array.
   */
  List<Tensor> execute(NamedTensorMap inputs, [List<String>? outputs]) {
    inputs = this._mapInputs(inputs);
    final names = inputs.keys.toList()..sort();
    this._checkInputs(inputs);
    this._checkInputShapeAndType(inputs);
    outputs = this._mapOutputs(outputs);
    this._checkOutputs(outputs);
    final inputNodes = names
        .map((name) => this.graph.nodes[parseNodeName(name).nodeName]!)
        .toList();
    final outputNodeNames =
        outputs.map((name) => parseNodeName(name).nodeName).toList();
    var outputNodes =
        outputNodeNames.map((name) => this.graph.nodes[name]!).toList();
    this._resetIntermediateTensors();
    // If no outputs are specified, then use the default outputs of the model.
    if (outputNodes.length == 0) {
      outputNodes = this._outputs;
    }

    final compilationKey = this._getCompilationKey(inputNodes, outputNodes);

    // Do nothing if the compiled graph cache contains the input.
    List<Node>? orderedNodes = this._compiledMap.get(compilationKey);
    if (orderedNodes == null) {
      orderedNodes = this._compile(inputs, outputNodes);
      this._compiledMap.set(compilationKey, orderedNodes);
    }

    final TensorArrayMap tensorArrayMap = {};
    final TensorListMap tensorListMap = {};

    return tidy(() {
      final context = ExecutionContext(this.weightMap, tensorArrayMap,
          tensorListMap, this.functionExecutorMap);
      final NamedTensorsMap tensorsMap = {...this.weightMap};

      inputs.keys.forEach((name) {
        final n = parseNodeName(name);
        tensorsMap[n.nodeName] = List.generate(
          n.index + 1,
          (index) => index == n.index ? inputs[name] : null,
        );
      });

      final tensorsToKeep = this._getFrozenTensorIds(tensorsMap);
      final intermediateTensorConsumerCount = <int, int>{};
      for (int i = 0; i < orderedNodes!.length; i++) {
        final node = orderedNodes[i];
        if (!tensorsMap.containsKey(node.name)) {
          final tensors =
              executeOp(node, tensorsMap, context, this._resourceManager);
          if (tensors is Future) {
            throw Exception(
                "The execution of the op '${node.op}' returned a promise. " +
                    "Please use model.executeAsync() instead.");
          }
          tensorsMap[node.name] = tensors;
          this._checkTensorForDisposal(node.name, node, tensorsMap, context,
              tensorsToKeep, outputNodeNames, intermediateTensorConsumerCount);
        }
      }
      // dispose the context for the root executor
      if (this.parent == null) {
        context.dispose(tensorsToKeep);
      }
      return outputs!
          .map((name) => getTensor(name, tensorsMap, context)!)
          .toList();
    });
  }

  Set<int> _getFrozenTensorIds(NamedTensorsMap tensorMap) {
    final ids = tensorMap.values
        .expand((tensors) => tensors.map((tensor) => tensor?.id))
        .whereType<int>()
        .toSet();
    return ids;
  }

  void _checkTensorForDisposal(
    String nodeName,
    Node node,
    NamedTensorsMap tensorMap,
    ExecutionContext context,
    Set<int> tensorsToKeep,
    List<String> outputNames,
    Map<int, int> intermediateTensorConsumerCount,
  ) {
    // Skip output nodes and any control flow nodes, since its dependency is
    // tricky to track correctly.
    if (node.category == 'control' || outputNames.indexOf(nodeName) != -1) {
      return;
    }

    tensorMap[nodeName]!.forEach((tensor) {
      if (tensor != null) {
        intermediateTensorConsumerCount[tensor.id] =
            (intermediateTensorConsumerCount[tensor.id] ?? 0) +
                node.children.length;
      }
    });
    node.inputs.forEach((input) {
      // Skip any control flow nodes, since its dependency is tricky to track
      // correctly.
      if (input.category != 'control') {
        final tensors =
            getTensorsForCurrentContenxt(input.name, tensorMap, context);
        if (tensors != null) {
          tensors.forEach((tensor) {
            if (tensor != null &&
                !tensor.kept &&
                !tensorsToKeep.contains(tensor.id)) {
              final count = intermediateTensorConsumerCount[tensor.id];
              if (count == 1) {
                if (!this._keepTensorForDebug) {
                  tensor.dispose();
                } else {
                  final nodeInfo = getNodeNameAndIndex(node.name, context);
                  nodeName = nodeInfo.nodeName;
                  final index = nodeInfo.index;
                  if (this._intermediateTensors.containsKey(nodeName)) {
                    this._intermediateTensors[nodeName]![index] = tensor;
                  } else {
                    this._intermediateTensors[nodeName] = [];
                    this._intermediateTensors[nodeName]![index] = tensor;
                  }
                }
                intermediateTensorConsumerCount.remove(tensor.id);
              } else if (count != null) {
                // only intermediate nodes has count set, inputs and weights are
                // not.
                intermediateTensorConsumerCount[tensor.id] =
                    intermediateTensorConsumerCount[tensor.id]! - 1;
              }
            }
          });
        }
      }
    });
  }

  /**
   * Executes the inference for given input tensors in Async fashion.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   */
  Future<List<Tensor>> executeAsync(
    NamedTensorMap inputs, [
    List<String>? outputs,
  ]) async {
    return this._executeAsync(inputs, outputs ?? this.outputNodes);
  }

  disposeIntermediateTensors() {
    if (this._intermediateTensors == null) {
      return;
    }
    this
        ._intermediateTensors
        .values
        .forEach((tensors) => tensors.forEach((tensor) => tensor?.dispose()));
    this._disposeTensorsMap();
  }

  void _disposeTensorsMap() {
    if (this._tensorsMap == null) {
      return;
    }
    this._tensorsMap!.values.forEach((tensorArray) {
      tensorArray.forEach((tensor) {
        if (tensor != null &&
            !tensor.kept &&
            !tensor.isDisposed &&
            !this._keepIds.contains(tensor.id)) {
          tensor.dispose();
        }
      });
    });
  }

  NamedTensorsMap? getIntermediateTensors() {
    return this._tensorsMap;
  }

  void _resetIntermediateTensors() {
    for (final key in this._intermediateTensors.keys.toList()) {
      this._intermediateTensors[key]!.forEach((tensor) => tensor?.dispose());
      this._intermediateTensors.remove(key);
    }
  }

  /**
   * Executes the inference for given input tensors in Async fashion.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs Optional. output node name from the Tensorflow model,
   * if no outputs are specified, the default outputs of the model would be
   * used. You can inspect intermediate nodes of the model by adding them to the
   * outputs array.
   * @param isFunctionExecution Optional. Flag for executing a function.
   * @param tensorArrayMap Optional, global TensorArray map by id. Used for
   * function execution.
   * @param tensorArrayMap Optinal global TensorList map by id. Used for
   * function execution.
   */
  Future<List<Tensor>> _executeAsync(
    NamedTensorMap inputs,
    List<String> outputs, [
    bool isFunctionExecution = false,
    TensorArrayMap? tensorArrayMap,
    TensorListMap? tensorListMap,
  ]) async {
    if (!isFunctionExecution) {
      inputs = this._mapInputs(inputs);
      this._checkInputs(inputs);
      this._checkInputShapeAndType(inputs);
      outputs = this._mapOutputs(outputs);
      this._checkOutputs(outputs);
    }

    // For model debug.
    try {
      this._keepTensorForDebug = env().getBool('KEEP_INTERMEDIATE_TENSORS');
    } catch (e, s) {
      util.log.warning('$e $s');
    }
    this._resetIntermediateTensors();

    final context = ExecutionContext(this.weightMap, tensorArrayMap ?? {},
        tensorListMap ?? {}, this.functionExecutorMap);

    // Graph with control flow op requires runtime evaluation of the execution
    // order, while without control flow the execution order is pre-determined
    // in the compile method.
    this._tensorsMap = await this
        ._executeWithControlFlow(inputs, context, outputs, isFunctionExecution);
    final results = outputs
        .map((name) => getTensor(name, this._tensorsMap!, context)!)
        .toList();

    // dispose all the intermediate tensors
    final outputIds = results.map((t) => t.id);
    final inputIds = inputs.values.map((t) => t.id);
    this._keepIds = {...outputIds, ...inputIds, ...this.weightIds};
    if (!this._keepTensorForDebug) {
      this._disposeTensorsMap();
    }

    // dispose the context for the root executor
    if (this.parent == null) {
      context.dispose(this._keepIds);
    }

    return results;
  }

  Future<List<Tensor>> executeFunctionAsync(
    List<Tensor> inputs,
    TensorArrayMap tensorArrayMap,
    TensorListMap tensorListMap,
  ) async {
    int index = 0;
    final NamedTensorMap mappedInputs = inputs.fold({}, (map, tensor) {
      map[this.inputs[index++].name] = tensor;
      return map;
    });

    return this._executeAsync(
        mappedInputs, this.outputNodes, true, tensorArrayMap, tensorListMap);
  }

  /**
   * When there are control flow nodes in the graph, the graph execution use
   * ExecutionContext to keep track of the frames and loop iterators.
   * @param inputs placeholder tensors for the graph.
   * @param context the execution context object for current execution.
   * @param outputNames Optional. output node name from the Tensorflow model,
   * if no outputs are specified, the default outputs of the model would be
   * used. You can inspect intermediate nodes of the model by adding them to the
   * outputs array.
   * @param isFunctionExecution Flag for executing a function.
   */
  Future<NamedTensorsMap> _executeWithControlFlow(
    NamedTensorMap inputs,
    ExecutionContext context, [
    List<String>? outputNames,
    bool? isFunctionExecution,
  ]) async {
    final names = inputs.keys;
    final inputNodes = names
        .map((name) => this.graph.nodes[parseNodeName(name).nodeName]!)
        .toList();
    final outputNodeNames = (outputNames ?? this.outputNodes)
        .map((name) => parseNodeName(name).nodeName)
        .toList();
    var outputNodes =
        outputNodeNames.map((name) => this.graph.nodes[name]!).toList();

    // If no outputs are specified, then use the default outputs of the model.
    if (outputNodes.length == 0) {
      outputNodes = this._outputs;
    }
    final execGraph = getExecutionSubgraph(
      inputs,
      outputNodes,
      this.weightMap,
      this._initNodes,
    );
    final usedNodes = execGraph.usedNodes;
    final missingInputs = execGraph.missingInputs;
    final dynamicNode = execGraph.dynamicNode;
    final syncInputs = execGraph.syncInputs;

    // First nodes to execute include inputNodes, weights, and initNodes.
    final List<NodeWithContexts> stack = [
      ...inputNodes,
      ...this.graph.weights,
      ...(this._initNodes ?? [])
    ].map((node) {
      return NodeWithContexts(node: node, contexts: context.currentContext);
    }).toList();
    final NamedTensorsMap tensorsMap = {...this.weightMap};
    inputs.keys.forEach((name) {
      final node = parseNodeName(name);
      final tensors = List.generate(
        node.index + 1,
        (index) => index == node.index ? inputs[name] : null,
      );
      tensorsMap[node.nodeName] = tensors;
    });
    final intermediateTensorConsumerCount = <int, int>{};
    final tensorsToKeep = this._getFrozenTensorIds(tensorsMap);
    final added = <String>{};
    while (stack.length > 0) {
      final promises = this._processStack(
          inputNodes,
          stack,
          context,
          tensorsMap,
          added,
          tensorsToKeep,
          outputNodeNames,
          intermediateTensorConsumerCount,
          usedNodes);
      await Future.wait(promises);
    }
    if (dynamicNode == null && isFunctionExecution != true) {
      util.log.warning(
          "This model execution did not contain any nodes with control flow " +
              "or dynamic output shapes. You can use model.execute() instead.");
    }
    final missingOutputs = outputNodes
        .where((node) =>
            !isControlFlow(node) &&
            getTensor(node.name, tensorsMap, context) == null)
        .map((node) => node.name);
    if (missingOutputs.isNotEmpty) {
      String alternativeMsg = '';
      if (dynamicNode != null) {
        alternativeMsg =
            "Alternatively, to avoid the dynamic ops, use model.execute() " +
                "and specify the inputs [${syncInputs}]";
      }
      throw Exception(
          "Cannot compute the outputs [${missingOutputs}] from the provided " +
              "inputs [${names}]. Consider providing the following inputs: " +
              "[${missingInputs}]. ${alternativeMsg}");
    }
    return tensorsMap;
  }

  List<Future<List<Tensor?>>> _processStack(
    List<Node> inputNodes,
    List<NodeWithContexts> stack,
    ExecutionContext context,
    NamedTensorsMap tensorMap,
    Set<String> added,
    Set<int> tensorsToKeep,
    List<String> outputNames,
    Map<int, int> intermediateTensorConsumerCount,
    Set<String> usedNodes,
  ) {
    final List<Future<List<Tensor?>>> promises = [];
    while (stack.length > 0) {
      final item = stack.removeLast();
      context.currentContext = item.contexts;
      String nodeName = '';
      // The tensor of the Enter op with isConstant set should be set
      // in the parent scope, so it will be available as constant for the
      // whole loop.
      final _isConstant =
          getParamValue('isConstant', item.node, tensorMap, context);
      if (item.node.op == 'Enter' &&
          _isConstant != null &&
          (_isConstant == true || (_isConstant is int && _isConstant != 0))) {
        nodeName = getNodeNameAndIndex(item.node.name, context).nodeName;
      }

      // only process nodes that are not in the tensorMap yet, this include
      // inputNodes and internal initNodes.
      if (tensorMap[item.node.name] == null) {
        final tensors =
            executeOp(item.node, tensorMap, context, this._resourceManager);
        if (nodeName == null || nodeName.isEmpty) {
          nodeName = getNodeNameAndIndex(item.node.name, context).nodeName;
        }
        final currentContext = context.currentContext;
        if (tensors is Future<List<Tensor?>>) {
          promises.add(tensors.then((t) {
            tensorMap[nodeName] = t;
            context.currentContext = currentContext;
            this._checkTensorForDisposal(
                nodeName,
                item.node,
                tensorMap,
                context,
                tensorsToKeep,
                outputNames,
                intermediateTensorConsumerCount);
            this._processChildNodes(
                item.node, stack, context, tensorMap, added, usedNodes);
            return t;
          }));
        } else {
          tensorMap[nodeName] = tensors;
          this._checkTensorForDisposal(nodeName, item.node, tensorMap, context,
              tensorsToKeep, outputNames, intermediateTensorConsumerCount);
          this._processChildNodes(
              item.node, stack, context, tensorMap, added, usedNodes);
        }
      } else {
        this._processChildNodes(
            item.node, stack, context, tensorMap, added, usedNodes);
      }
    }
    return promises;
  }

  void _processChildNodes(
    Node node,
    List<NodeWithContexts> stack,
    ExecutionContext context,
    NamedTensorsMap tensorMap,
    Set<String> added,
    Set<String> usedNodes,
  ) {
    node.children.forEach((childNode) {
      final nodeName = getNodeNameAndIndex(childNode.name, context).nodeName;
      if (added.contains(nodeName) || !usedNodes.contains(childNode.name)) {
        return;
      }
      // Merge op can be pushed if any of its inputs has value.
      if (childNode.op == 'Merge') {
        if (childNode.inputNames.any((name) {
          return getTensor(name, tensorMap, context) != null;
        })) {
          added.add(nodeName);
          stack.add(NodeWithContexts(
              contexts: context.currentContext, node: childNode));
        }
      } else // Otherwise all inputs must to have value.
      if (childNode.inputNames.every((name) {
        return getTensor(name, tensorMap, context) != null;
      })) {
        added.add(nodeName);
        stack.add(NodeWithContexts(
            contexts: context.currentContext, node: childNode));
      }
    });
  }

  /**
   * Releases the memory used by the weight tensors.
   */
  void dispose() {
    this
        .weightMap
        .values
        .forEach((tensors) => tensors.forEach((tensor) => tensor?.dispose()));
  }

  void _checkInputShapeAndType(NamedTensorMap inputs) {
    inputs.keys.forEach((name) {
      final input = inputs[name]!;
      final nodeName = parseNodeName(name).nodeName;
      final node = this.graph.nodes[nodeName]!;
      if (node.attrParams['shape']?.value != null) {
        final shape = node.attrParams['shape']!.value as List<int>;
        int index = -1;
        final match = shape.length == input.shape.length &&
            input.shape
                .every((dim) => shape[++index] == -1 || shape[index] == dim);
        util.assert_(
            match,
            () =>
                "The shape of dict['${node.name}'] provided in " +
                "model.execute(dict) must be [${shape}], but was " +
                "[${input.shape}]");
      }
      if (node.attrParams['dtype']?.value != null) {
        util.assert_(
            input.dtype == node.attrParams['dtype']!.value as String,
            () =>
                "The dtype of dict['${node.name}'] provided in " +
                "model.execute(dict) must be " +
                "${node.attrParams['dtype']!.value}, but was ${input.dtype}");
      }
    });
  }

  NamedTensorMap _mapInputs(NamedTensorMap inputs) {
    final NamedTensorMap result = {};
    for (final inputName in inputs.keys) {
      final tensor = this._signature?.inputs?[inputName];
      if (tensor != null) {
        result[tensor.name!] = inputs[inputName]!;
      } else {
        result[inputName] = inputs[inputName]!;
      }
    }
    return result;
  }

  void _checkInputs(NamedTensorMap inputs) {
    final notInGraph = inputs.keys.where((name) {
      final nodeName = parseNodeName(name).nodeName;
      return this.graph.nodes[nodeName] == null;
    });
    if (notInGraph.isNotEmpty) {
      throw Exception("The dict provided in model.execute(dict) has " +
          "keys: [${notInGraph}] that are not part of graph");
    }
  }

  List<String> _mapOutputs(List<String>? outputs) {
    if (outputs == null) return this.outputNodes;
    return outputs.map((name) {
      final tensor = this._signature?.outputs?[name];
      if (tensor != null) {
        return tensor.name!;
      }
      return name;
    }).toList();
  }

  void _checkOutputs(List<String> outputs) {
    outputs.forEach((name) {
      final normalizedName = parseNodeName(name).nodeName;
      if (!this.graph.nodes.containsKey(normalizedName)) {
        throw Exception("The output '${name}' is not found in the graph");
      }
    });
  }
}
