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
import 'dart:async';

import 'package:tensorflow_wasm/src/converter/executor/execution_context.dart';
import 'package:tensorflow_wasm/src/converter/executor/resource_manager.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/converter/data/compiled_api.dart'
    as tensorflow;

// import {Tensor} from '@tensorflow/tfjs-core';

// import * as tensorflow from '../data/compiled_api';
// import {NamedTensorsMap} from '../data/types';
// import {ExecutionContext} from '../executor/execution_context';
// import {ResourceManager} from '../executor/resource_manager';

/// 'number'|'string'|'string[]'|'number[]'|'bool'|'bool[]'|
///     'shape'|'shape[]'|'tensor'|'tensors'|'dtype'|'dtype[]'|'func';
typedef ParamType = String;

/// 'arithmetic'|'basic_math'|'control'|'convolution'|
///     'creation'|'custom'|'dynamic'|'evaluation'|'graph'|'hash_table'|'image'|
///     'logical'|'matrices'|'normalization'|'reduction'|'slice_join'|'sparse'|
///     'spectral'|'string'|'transformation'
typedef Category = String;

// For mapping input or attributes of NodeDef into TensorFlow.js op param.
class ParamMapper {
  // tensorflow.js name for the field, it should be in camelcase format.
  final String name;
  final ParamType type;
  final ValueType? defaultValue;
  final bool? notSupported;

  const ParamMapper({
    required this.name,
    required this.type,
    this.defaultValue,
    this.notSupported,
  });

  const factory ParamMapper.fromModel(ParamMapper mapper) = _ParamMapperWrap;
}

class _ParamMapperWrap implements ParamMapper {
  const _ParamMapperWrap(this.mapper);
  final ParamMapper mapper;

  @override
  ValueType? get defaultValue => mapper.defaultValue;

  @override
  String get name => mapper.name;

  @override
  bool? get notSupported => mapper.notSupported;

  @override
  ParamType get type => mapper.type;
}

// For mapping the input of TensorFlow NodeDef into TensorFlow.js Op param.
class InputParamMapper extends _ParamMapperWrap {
  // The first number is the starting index of the param, the second number is
  // the length of the param. If the length value is positive number, it
  // represents the true length of the param. Otherwise, it represents a
  // variable length, the value is the index go backward from the end of the
  // array.
  // For example `[0, 5]`: this param is the array of input tensors starting at
  // index 0 and with the length of 5.
  // For example `[1, -1]`: this param is the array of input tensors starting at
  // index 1 and with the `inputs.length - 1`.
  // Zero-based index at where in the input array this param starts.
  // A negative index can be used, indicating an offset from the end of the
  // sequence. slice(-2) extracts the last two elements in the sequence.
  final int start;
  // Zero-based index before where in the input array the param ends. The
  // mapping is up to but not including end. For example, start = 1, end = 4
  // includes the second element through the fourth element (elements indexed 1,
  // 2, and 3). A negative index can be used, indicating an offset from the end
  // of the sequence. start = 2, end = -1 includes the third element through the
  // second-to-last element in the sequence. If end is omitted, end is set to
  // start + 1, the mapping only include the single element at start index. If
  // end is set to 0, the mapping is through the end of the input array
  // (arr.length). If end is greater than the length of the inputs, mapping
  // inncludes through to the end of the sequence (arr.length).
  final int? end;

  const InputParamMapper({
    required this.start,
    this.end,
    required ParamMapper mapper,
  }) : super(mapper);
}

// For mapping the attributes of TensorFlow NodeDef into TensorFlow.js op param.
class AttrParamMapper extends _ParamMapperWrap {
  // TensorFlow attribute name, this should be set if the tensorflow attribute
  // name is different form the tensorflow.js name.
  final String? tfName;
  // TensorFlow deprecated attribute name, this is used to support old models.
  final String? tfDeprecatedName;

  const AttrParamMapper({
    this.tfName,
    this.tfDeprecatedName,
    required ParamMapper mapper,
  }) : super(mapper);
}

abstract class InternalOpExecutor {
  // TODO: was Tensors
  List<Tensor> call(
    Node node,
    NamedTensorsMap tensorMap,
    ExecutionContext context,
  );
}

abstract class InternalOpAsyncExecutor {
  Future<List<Tensor>> call(
    Node node,
    NamedTensorsMap tensorMap,
    ExecutionContext context,
    ResourceManager? resourceManager,
  );
}

class OpMapper {
  final String tfOpName;
  final Category? category;
  final List<InputParamMapper>? inputs;
  final List<AttrParamMapper>? attrs;
  final List<String>? outputs;
  final OpExecutor? customExecutor;

  const OpMapper({
    required this.tfOpName,
    this.category,
    this.inputs,
    this.attrs,
    this.outputs,
    this.customExecutor,
  });
}

class Node {
  String? signatureKey;
  final String name;
  final String op;
  final Category category;
  final List<String> inputNames;
  final List<Node> inputs;
  final Map<String, InputParamValue> inputParams;
  final Map<String, ParamValue> attrParams;
  final List<Node> children;
  final Map<String, tensorflow.IAttrValue>? rawAttrs;
  int? defaultOutput;
  final List<String>? outputs;

  Node({
    this.signatureKey,
    required this.name,
    required this.op,
    required this.category,
    required this.inputNames,
    required this.inputs,
    required this.inputParams,
    required this.attrParams,
    required this.children,
    this.rawAttrs,
    this.defaultOutput,
    this.outputs,
  });
}

class Graph {
  final Map<String, Node> nodes;
  final List<Node> placeholders;
  final List<Node> inputs;
  final List<Node> outputs;
  final List<Node> weights;
  final tensorflow.ISignatureDef? signature;
  final Map<String, Graph>? functions;
  final List<Node>? initNodes;

  Graph({
    required this.nodes,
    required this.placeholders,
    required this.inputs,
    required this.outputs,
    required this.weights,
    this.signature,
    this.functions,
    this.initNodes,
  });
}

/// String|List<String>|number|List<number>|List<number>[]|boolean|boolean[]|Tensor|Tensor[]
typedef ValueType = Object;

class ParamValue {
  final ValueType? value;
  final ParamType type;

  ParamValue({
    this.value,
    required this.type,
  });
}

class InputParamValue implements ParamValue {
  final int? inputIndexStart;
  final int? inputIndexEnd;
  final ValueType? value;
  final ParamType type;

  InputParamValue({
    this.inputIndexStart,
    this.inputIndexEnd,
    this.value,
    required this.type,
  });
}

abstract class OpExecutor {
  FutureOr<Tensors> call(GraphNode node);
}

class GraphNode {
  final List<Tensor> inputs;
  final Map<String, ValueType?> attrs;

  GraphNode({
    required this.inputs,
    required this.attrs,
  });
}
