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

// import {DataType, env} from '@tensorflow/tfjs-core';

// import * as tensorflow from '../data/compiled_api';

// import {getRegisteredOp} from './custom_op/register';
// import {getNodeNameAndIndex} from './executors/utils';
import './op_list/arithmetic.dart' as arithmetic;
import './op_list/basic_math.dart' as basicMath;
import './op_list/control.dart' as control;
import './op_list/convolution.dart' as convolution;
import './op_list/creation.dart' as creation;
import './op_list/dynamic.dart' as dynamic_;
import './op_list/evaluation.dart' as evaluation;
import './op_list/graph.dart' as graph;
import './op_list/hash_table.dart' as hashTable;
import './op_list/image.dart' as image;
import './op_list/logical.dart' as logical;
import './op_list/matrices.dart' as matrices;
import './op_list/normalization.dart' as normalization;
import './op_list/reduction.dart' as reduction;
import './op_list/slice_join.dart' as sliceJoin;
import './op_list/sparse.dart' as sparse;
import './op_list/spectral.dart' as spectral;
import './op_list/string.dart' as string;
import './op_list/transformation.dart' as transformation;
// import {Graph, InputParamValue, Node, OpMapper, ParamValue} from './types';

import 'dart:convert' show base64, utf8;

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/src/tensor.dart';

import 'custom_op/register.dart';
import 'executors/utils.dart';
import 'types.dart';
import '../data/compiled_api.dart' as tensorflow;

class OperationMapper {
  static OperationMapper? _instance;

  late final Map<String, OpMapper> opMappers;

  // Singleton instance for the mapper
  static OperationMapper get Instance {
    return _instance ?? (_instance = OperationMapper._());
  }

  // Loads the op mapping from the JSON file.
  OperationMapper._() {
    const ops = [
      arithmetic.opMappers,
      basicMath.opMappers,
      control.opMappers,
      convolution.opMappers,
      creation.opMappers,
      dynamic_.opMappers,
      evaluation.opMappers,
      graph.opMappers,
      hashTable.opMappers,
      image.opMappers,
      logical.opMappers,
      matrices.opMappers,
      normalization.opMappers,
      reduction.opMappers,
      sliceJoin.opMappers,
      sparse.opMappers,
      spectral.opMappers,
      string.opMappers,
      transformation.opMappers,
    ];
    final List<OpMapper> mappersJson = [...ops.expand((op) => op)];

    this.opMappers = mappersJson.fold<Map<String, OpMapper>>({}, (map, mapper) {
      map[mapper.tfOpName] = mapper;
      return map;
    });
  }

  // Converts the model inference graph from Tensorflow GraphDef to local
  // representation for TensorFlow.js API
  Graph transformGraph(
    tensorflow.IGraphDef graph, [
    tensorflow.ISignatureDef? signature,
  ]) {
    final tfNodes = graph.node!;
    final List<Node> placeholders = [];
    final List<Node> weights = [];
    final List<Node> initNodes = [];
    final nodes = tfNodes.fold<Map<String, Node>>({}, (map, node) {
      final nodeName = node.name!;
      final n = this._mapNode(node);
      map[nodeName] = n;
      if (node.op?.startsWith('Placeholder') == true) {
        placeholders.add(n);
      } else if (node.op == 'Const') {
        weights.add(n);
      } else if (node.input == null || node.input!.length == 0) {
        initNodes.add(n);
      }
      return map;
    });

    List<Node> inputs = [];
    final List<Node> outputs = [];
    Map<String, String> inputNodeNameToKey = {};
    Map<String, String> outputNodeNameToKey = {};
    if (signature != null) {
      inputNodeNameToKey = this._mapSignatureEntries(signature.inputs);
      outputNodeNameToKey = this._mapSignatureEntries(signature.outputs);
    }
    final allNodes = nodes.keys;
    allNodes.forEach((key) {
      final node = nodes[key]!;
      node.inputNames.forEachIndexed((index, name) {
        final n = getNodeNameAndIndex(name);
        final nodeName = n.nodeName;
        final inputNode = nodes[nodeName]!;
        if (inputNode.outputs != null) {
          final outputIndex = inputNode.outputs!.indexOf(n.outputName!);
          if (outputIndex != -1) {
            final inputName = '${nodeName}:${outputIndex}';
            // update the input name to use the mapped output index directly.
            node.inputNames[index] = inputName;
          }
        }
        node.inputs.add(inputNode);
        inputNode.children.add(node);
      });
    });

    // if signature has not outputs set, add any node that does not have
    // outputs.
    if (outputNodeNameToKey.length == 0) {
      allNodes.forEach((key) {
        final node = nodes[key]!;
        if (node.children.length == 0) {
          outputs.add(node);
        }
      });
    } else {
      outputNodeNameToKey.keys.forEach((name) {
        final n = getNodeNameAndIndex(name);
        final node = nodes[n.nodeName];
        if (node != null) {
          node.signatureKey = outputNodeNameToKey[name];
          outputs.add(node);
        }
      });
    }

    if (inputNodeNameToKey.length > 0) {
      inputNodeNameToKey.keys.forEach((name) {
        final n = getNodeNameAndIndex(name);
        final node = nodes[n.nodeName];
        if (node != null) {
          node.signatureKey = inputNodeNameToKey[name];
          inputs.add(node);
        }
      });
    } else {
      inputs = placeholders;
    }

    Map<String, Graph> functions = {};
    if (graph.library?.function != null) {
      functions = graph.library!.function!.fold<Map<String, Graph>>({},
          (functions, func) {
        functions[func.signature!.name!] = this._mapFunction(func);
        return functions;
      });
    }

    final result = Graph(
      nodes: nodes,
      inputs: inputs,
      outputs: outputs,
      weights: weights,
      placeholders: placeholders,
      signature: signature,
      functions: functions,
      initNodes: initNodes.isNotEmpty ? initNodes : null,
    );

    return result;
  }

  Map<String, String> _mapSignatureEntries(
      Map<String, tensorflow.ITensorInfo>? entries) {
    return (entries ?? {}).entries.fold<Map<String, String>>({}, (prev, entry) {
      prev[entry.value.name!] = entry.key;
      return prev;
    });
  }

  Node _mapNode(tensorflow.INodeDef node) {
    // Unsupported ops will cause an error at run-time (not parse time), since
    // they may not be used by the actual execution subgraph.
    final mapper =
        getRegisteredOp(node.op!) ?? this.opMappers[node.op] ?? {} as OpMapper;

    node.attr ??= {};

    final attr = node.attr!;

    final newNode = Node(
      name: node.name!,
      op: node.op!,
      category: mapper.category!,
      inputNames: (node.input ?? [])
          .map((input) => input.startsWith('^') ? input.substring(1) : input)
          .toList(),
      inputs: [],
      children: [],
      rawAttrs: node.attr,
      outputs: mapper.outputs,
      inputParams: mapper.inputs == null
          ? {}
          : mapper.inputs!.fold<Map<String, InputParamValue>>({}, (map, param) {
              map[param.name] = InputParamValue(
                  type: param.type,
                  inputIndexStart: param.start,
                  inputIndexEnd: param.end);
              return map;
            }),
      attrParams: mapper.attrs == null
          ? {}
          : mapper.attrs!.fold<Map<String, ParamValue>>({}, (map, param) {
              final type = param.type;
              final tfName = param.tfName!;
              Object? value;
              switch (param.type) {
                case 'string':
                  value = getStringParam(
                      attr, tfName, param.defaultValue as String);

                  if (value == null && param.tfDeprecatedName != null) {
                    value = getStringParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as String);
                  }
                  break;
                case 'string[]':
                  value = getStringArrayParam(
                      attr, tfName, param.defaultValue as List<String>);

                  if (value == null && param.tfDeprecatedName != null) {
                    value = getStringArrayParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as List<String>);
                  }
                  break;
                case 'number':
                  value = getNumberParam(
                      attr, tfName, (param.defaultValue ?? 0) as int);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getNumberParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as int);
                  }
                  break;
                case 'number[]':
                  value = getNumericArrayParam(
                      attr, tfName, param.defaultValue as List<num>);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getNumericArrayParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as List<num>);
                  }
                  break;
                case 'bool':
                  value =
                      getBoolParam(attr, tfName, param.defaultValue as bool);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getBoolParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as bool);
                  }
                  break;
                case 'bool[]':
                  value = getBoolArrayParam(
                      attr, tfName, param.defaultValue as List<bool>);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getBoolArrayParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as List<bool>);
                  }
                  break;
                case 'shape':
                  value = getTensorShapeParam(
                      attr, tfName, param.defaultValue as List<int>);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getTensorShapeParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as List<int>);
                  }
                  break;
                case 'shape[]':
                  value = getTensorShapeArrayParam(
                      attr, tfName, param.defaultValue as List<List<int>>);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getTensorShapeArrayParam(
                        attr,
                        param.tfDeprecatedName!,
                        param.defaultValue as List<List<int>>);
                  }
                  break;
                case 'dtype':
                  value = getDtypeParam(
                      attr, tfName, param.defaultValue as DataType);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getDtypeParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as DataType);
                  }
                  break;
                case 'dtype[]':
                  value = getDtypeArrayParam(
                      attr, tfName, param.defaultValue as List<DataType>);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getDtypeArrayParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as List<DataType>);
                  }
                  break;
                case 'func':
                  value =
                      getFuncParam(attr, tfName, param.defaultValue as String);
                  if (value == null && param.tfDeprecatedName != null) {
                    value = getFuncParam(attr, param.tfDeprecatedName!,
                        param.defaultValue as String);
                  }
                  break;
                case 'tensor':
                case 'tensors':
                  break;
                default:
                  throw Exception(
                      'Unsupported param type: ${param.type} for op: ${node.op}');
              }
              map[param.name] = ParamValue(value: value, type: type);
              return map;
            }),
    );
    return newNode;
  }

  // map the TFunctionDef to TFJS graph object
  Graph _mapFunction(tensorflow.IFunctionDef functionDef) {
    final tfNodes = functionDef.nodeDef;
    final List<Node> placeholders = [];
    final List<Node> weights = [];
    Map<String, Node> nodes = {};
    if (tfNodes != null) {
      nodes = tfNodes.fold<Map<String, Node>>({}, (map, node) {
        map[node.name!] = this._mapNode(node);
        if (node.op == 'Const') {
          weights.add(map[node.name!]!);
        }
        return map;
      });
    }
    final List<Node> inputs = [];
    final List<Node> outputs = [];

    functionDef.signature!.inputArg!.forEach((arg) {
      final n = getNodeNameAndIndex(arg.name!);
      final node = Node(
        name: n.nodeName,
        op: 'Placeholder',
        inputs: [],
        inputNames: [],
        category: 'graph',
        inputParams: {},
        attrParams: {
          'dtype': ParamValue(value: parseDtypeParam(arg.type!), type: 'dtype')
        },
        children: [],
      );
      node.signatureKey = arg.name;
      inputs.add(node);
      nodes[n.nodeName] = node;
    });

    nodes.keys.forEach((key) {
      final node = nodes[key]!;
      node.inputNames.forEachIndexed((index, name) {
        final n = getNodeNameAndIndex(name);
        final nodeName = n.nodeName;
        final inputNode = nodes[nodeName]!;
        if (inputNode.outputs != null) {
          final outputIndex = inputNode.outputs!.indexOf(n.outputName!);
          if (outputIndex != -1) {
            final inputName = '${nodeName}:${outputIndex}';
            // update the input name to use the mapped output index directly.
            node.inputNames[index] = inputName;
          }
        }
        node.inputs.add(inputNode);
        inputNode.children.add(node);
      });
    });

    final returnNodeMap = functionDef.ret;

    functionDef.signature!.outputArg!.forEach((output) {
      final n = getNodeNameAndIndex(returnNodeMap![output.name]!);
      final node = nodes[n.nodeName];
      if (node != null) {
        node.defaultOutput = n.index;
        outputs.add(node);
      }
    });

    final signature = this._mapArgsToSignature(functionDef);
    return Graph(
      nodes: nodes,
      inputs: inputs,
      outputs: outputs,
      weights: weights,
      placeholders: placeholders,
      signature: signature,
    );
  }

  tensorflow.ISignatureDef _mapArgsToSignature(
      tensorflow.IFunctionDef functionDef) {
    return tensorflow.ISignatureDef(
      methodName: functionDef.signature?.name,
      inputs: functionDef.signature!.inputArg!
          .fold<Map<String, tensorflow.ITensorInfo>>({}, (map, arg) {
        map[arg.name!] = this._mapArgToTensorInfo(arg, null);
        return map;
      }),
      outputs: functionDef.signature!.outputArg!
          .fold<Map<String, tensorflow.ITensorInfo>>({}, (map, arg) {
        map[arg.name!] = this._mapArgToTensorInfo(arg, functionDef.ret);
        return map;
      }),
    );
  }

  tensorflow.ITensorInfo _mapArgToTensorInfo(
      tensorflow.OpDef_IArgDef arg, Map<String, String>? nameMap) {
    var name = arg.name;
    if (nameMap != null) {
      name = nameMap[name];
    }
    return tensorflow.ITensorInfo(name: name, dtype: arg.type);
  }
}

String decodeBase64(String text) {
  return utf8.decode(base64.decode(text));
  // final global = env().global;
  // if (typeof global.atob != 'undefined') {
  //   return global.atob(text);
  // } else if (typeof Buffer != 'undefined') {
  //   return new Buffer(text, 'base64').toString();
  // } else {
  //   throw Exception(
  //       'Unable to decode base64 in this environment. ' +
  //       'Missing built-in atob() or Buffer()');
  // }
}

String parseStringParam(
  String s, //: []|string
  bool keepCase,
) {
  final value =
      s is List ? String.fromCharCodes(s as List<int>) : decodeBase64(s);
  return keepCase ? value : value.toLowerCase();
}

String getStringParam(
  Map<String, tensorflow.IAttrValue> attrs,
  String name,
  String def, {
  bool keepCase = false,
}) {
  final param = attrs[name];
  if (param?.s != null) {
    return parseStringParam(param!.s!, keepCase);
  }
  return def;
}

bool getBoolParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, bool def) {
  final param = attrs[name];
  return param?.b ?? def;
}

int getNumberParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, int def) {
  final param = attrs[name];
  final value = param?.i ?? param?.f ?? def;
  return value is int ? value : int.parse(value as String);
}

DataType? parseDtypeParam(tensorflow.DataType value) {
  if (value is String) {
    // tslint:disable-next-line:no-any
    value = tensorflow.DataType.values.byName(value as String);
  }
  switch (value) {
    case tensorflow.DataType.DT_FLOAT:
    case tensorflow.DataType.DT_HALF:
      return 'float32';
    case tensorflow.DataType.DT_INT32:
    case tensorflow.DataType.DT_INT64:
    case tensorflow.DataType.DT_INT8:
    case tensorflow.DataType.DT_UINT8:
      return 'int32';
    case tensorflow.DataType.DT_BOOL:
      return 'bool';
    case tensorflow.DataType.DT_DOUBLE:
      return 'float32';
    case tensorflow.DataType.DT_STRING:
      return 'string';
    default:
      // Unknown dtype error will happen at runtime (instead of parse time),
      // since these nodes might not be used by the actual subgraph execution.
      return null;
  }
}

String getFuncParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, String def) {
  final param = attrs[name];
  return param?.func?.name ?? def;
}

DataType? getDtypeParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, DataType def) {
  final param = attrs[name];
  if (param?.type != null) {
    return parseDtypeParam(param!.type!);
  }
  return def;
}

List<DataType> getDtypeArrayParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, List<DataType> def) {
  final param = attrs[name];
  if (param?.list?.type != null) {
    return param!.list!.type!.map((v) => parseDtypeParam(v)!).toList();
  }
  return def;
}

List<int>? parseTensorShapeParam(tensorflow.ITensorShape shape) {
  if (shape.unknownRank == true) {
    return null;
  }
  if (shape.dim != null) {
    return shape.dim!
        .map<int>((dim) =>
            (dim.size is int) ? dim.size as int : int.parse(dim.size as String))
        .toList();
  }
  return [];
}

List<int>? getTensorShapeParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, List<int>? def) {
  final param = attrs[name];
  if (param != null && param.shape != null) {
    return parseTensorShapeParam(param.shape!);
  }
  return def;
}

List<num> getNumericArrayParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, List<num> def) {
  final paramList = attrs[name]?.list;
  if (paramList != null) {
    return ((paramList.f != null && paramList.f!.isNotEmpty
                ? paramList.f
                : paramList.i) ??
            [])
        .map<num>((v) => v is num ? v : int.parse(v as String))
        .toList();
  }
  return def;
}

List<String> getStringArrayParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, List<String> def,
    {bool keepCase = false}) {
  final param = attrs[name];
  if (param?.list?.s != null) {
    return param!.list!.s!.map((v) {
      return parseStringParam(v, keepCase);
    }).toList();
  }
  return def;
}

List<List<int>> getTensorShapeArrayParam(
    Map<String, tensorflow.IAttrValue> attrs,
    String name,
    List<List<int>> def) {
  final param = attrs[name];
  return param?.list?.shape?.map((v) {
        return parseTensorShapeParam(v)!;
      }).toList() ??
      def;
}

List<bool> getBoolArrayParam(
    Map<String, tensorflow.IAttrValue> attrs, String name, List<bool> def) {
  final param = attrs[name];
  return param?.list?.b ?? def;
}
