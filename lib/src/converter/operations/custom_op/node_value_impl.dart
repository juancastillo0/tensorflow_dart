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

// import {DataType, Tensor} from '@tensorflow/tfjs-core';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {getTensor} from '../executors/utils';
// import {getBoolArrayParam, getBoolParam, getDtypeArrayParam, getDtypeParam, getNumberParam, getNumericArrayParam, getStringArrayParam, getStringParam, getTensorShapeArrayParam, getTensorShapeParam} from '../operation_mapper';
// import {GraphNode, Node, ValueType} from '../types';

import 'package:tensorflow_wasm/src/tensor.dart';

import '../../executor/execution_context.dart';
import '../executors/utils.dart';
import '../operation_mapper.dart';
import '../types.dart';

/**
 * Helper class for lookup inputs and params for nodes in the model graph.
 */
class NodeValueImpl implements GraphNode {
  late final List<Tensor> inputs;
  late final Map<String, ValueType?> attrs;

  // private
  final Node node;
  final NamedTensorsMap tensorMap;
  final ExecutionContext context;

  NodeValueImpl(
    this.node,
    this.tensorMap,
    this.context,
  ) {
    this.inputs = node.inputNames.map((name) => this._getInput(name)!).toList();
    if (node.rawAttrs != null) {
      this.attrs =
          node.rawAttrs!.keys.fold<Map<String, ValueType?>>({}, (attrs, key) {
        attrs[key] = this._getAttr(key);
        return attrs;
      });
    } else {
      this.attrs = {};
    }
  }

  /**
   * Return the value of the attribute or input param.
   * @param name String: name of attribute or input param.
   */
  Tensor? _getInput(String name) {
    return getTensor(name, this.tensorMap, this.context);
  }

  /**
   * Return the value of the attribute or input param.
   * @param name String: name of attribute or input param.
   */
  ValueType? _getAttr(String name, [ValueType? defaultValue]) {
    final value = this.node.rawAttrs![name]!;
    final rawAttrs = this.node.rawAttrs!;
    if (value.tensor != null) {
      return getTensor(name, this.tensorMap, this.context);
    }
    if (value.i != null || value.f != null) {
      return getNumberParam(rawAttrs, name, defaultValue as int?);
    }
    if (value.s != null) {
      return getStringParam(rawAttrs, name, defaultValue as String?);
    }
    if (value.b != null) {
      return getBoolParam(rawAttrs, name, defaultValue as bool?);
    }
    if (value.shape != null) {
      return getTensorShapeParam(rawAttrs, name, defaultValue as List<int>?);
    }
    if (value.type != null) {
      return getDtypeParam(rawAttrs, name, defaultValue as DataType?);
    }
    if (value.list != null) {
      final list = value.list!;
      if (list.i != null || list.f != null) {
        return getNumericArrayParam(rawAttrs, name, defaultValue as List<num>?);
      }
      if (list.s != null) {
        return getStringArrayParam(
            rawAttrs, name, defaultValue as List<String>?);
      }
      if (list.shape != null) {
        return getTensorShapeArrayParam(
            rawAttrs, name, defaultValue as List<List<int>>?);
      }
      if (list.b != null) {
        return getBoolArrayParam(rawAttrs, name, defaultValue as List<bool>?);
      }
      if (list.type != null) {
        return getDtypeArrayParam(
            rawAttrs, name, defaultValue as List<DataType>?);
      }
    }

    return defaultValue;
  }
}
