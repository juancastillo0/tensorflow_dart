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
// import {Tensor} from '@tensorflow/tfjs-core';

// import {NamedTensorsMap, TensorArrayMap, TensorListMap} from '../data/types';

// import {TensorArray} from './tensor_array';
// import {TensorList} from './tensor_list';
// import {FunctionExecutor} from './types';

import 'package:tensorflow_wasm/src/converter/executor/resource_manager.dart';
import 'package:tensorflow_wasm/src/converter/executor/tensor_array.dart';
import 'package:tensorflow_wasm/src/converter/executor/tensor_list.dart';
import 'package:tensorflow_wasm/src/tensor.dart' hide TensorList;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' show SliceList;

abstract class FunctionExecutor {
  Future<List<Tensor>> executeFunctionAsync(
    List<Tensor> inputs,
    TensorArrayMap tensorArrayMap,
    TensorListMap tensorListMap,
  );
  NamedTensorsMap get weightMap;
}

class ExecutionContextInfo {
  final int id; // the unique id of the context info
  final String frameName; // The frame name of the loop, this comes from
  // the TensorFlow NodeDef.
  final int iterationId; // The iteration id of the loop

  ExecutionContextInfo({
    required this.id,
    required this.frameName,
    required this.iterationId,
  });

  ExecutionContextInfo copyWith({
    int? id,
    String? frameName,
    int? iterationId,
  }) =>
      ExecutionContextInfo(
        id: id ?? this.id,
        frameName: frameName ?? this.frameName,
        iterationId: iterationId ?? this.iterationId,
      );
}

/**
 * ExecutionContext captures the runtime environment of the node. It keeps
 * track of the current frame and iteration for the control flow ops.
 *
 * For example, typical Dynamic RNN model may contain loops, for which
 * TensorFlow will generate graphs with Enter/Exit nodes to control the
 * current execution frame, and NextIteration Nodes for iteration id increment.
 * For model with branch logic, TensorFLow will generate Switch/Merge ops.
 */
class ExecutionContext {
  // private
  List<ExecutionContextInfo> _contexts = [
    ExecutionContextInfo(id: 0, frameName: '', iterationId: 0)
  ];
  int _lastId = 0;
  late final List<String> _currentContextIds;

  final NamedTensorsMap weightMap;
  final TensorArrayMap tensorArrayMap;
  final TensorListMap tensorListMap;
  final Map<String, FunctionExecutor> functionMap;

  ExecutionContext(
    this.weightMap,
    this.tensorArrayMap,
    this.tensorListMap,
    this.functionMap,
  ) {
    this._generateCurrentContextIds();
  }

  ExecutionContextInfo _newFrame(int id, String frameName) {
    return ExecutionContextInfo(id: id, frameName: frameName, iterationId: 0);
  }

  /**
   * Set the current context
   * @param contexts: ExecutionContextInfo[] the current path of execution
   * frames
   */
  set currentContext(contexts) {
    if (this._contexts != contexts) {
      this._contexts = contexts;
      this._generateCurrentContextIds();
    }
  }

  List<ExecutionContextInfo> get currentContext {
    return this._contexts;
  }

  /**
   * Returns the current context in string format.
   */
  String get currentContextId {
    return this._currentContextIds[0];
  }

  /**
   * Returns the current context and all parent contexts in string format.
   * This allow access to the nodes in the current and parent frames.
   */
  List<String> get currentContextIds {
    return this._currentContextIds;
  }

  void _generateCurrentContextIds() {
    final names = <String>[];
    for (int i = 0; i < this._contexts.length - 1; i++) {
      final contexts = this._contexts.sublistRelaxed(0, this._contexts.length - i);
      names.add(this._contextIdforContexts(contexts));
    }
    names.add('');
    this._currentContextIds = names;
  }

  String _contextIdforContexts(List<ExecutionContextInfo>? contexts) {
    return contexts != null
        ? contexts
            .map((context) => (context.id == 0 && context.iterationId == 0)
                ? ''
                : '${context.frameName}-${context.iterationId}')
            .join('/')
        : '';
  }

  /**
   * Enter a new frame, a new context is pushed on the current context list.
   * @param frameId new frame id
   */
  void enterFrame(String frameId) {
    if (this._contexts != null) {
      this._lastId++;
      this._contexts = this._contexts.sublist(0);
      this._contexts.add(this._newFrame(this._lastId, frameId));
      this
          ._currentContextIds
          .insert(0, this._contextIdforContexts(this._contexts));
    }
  }

  /**
   * Exit the current frame, the last context is removed from the current
   * context list.
   */
  void exitFrame() {
    if (this._contexts != null && this._contexts.length > 1) {
      this._contexts = this._contexts.sublist(0);
      this._contexts.removeLast();
      this.currentContextIds.removeAt(0);
    } else {
      throw Exception('Cannot exit frame, the context is empty');
    }
  }

  /**
   * Enter the next iteration of a loop, the iteration id of last context is
   * increased.
   */
  void nextIteration() {
    if (this._contexts != null && this._contexts.length > 0) {
      this._contexts = this._contexts.sublist(0);
      this._lastId++;
      final _baseContext = this._contexts[this._contexts.length - 1];
      final context = _baseContext.copyWith(
        id: this._lastId,
        iterationId: _baseContext.iterationId + 1,
      );
      this._contexts[this._contexts.length - 1] = context;
      this._currentContextIds[0] = this._contextIdforContexts(this._contexts);
    } else {
      throw Exception('Cannot increase frame iteration, the context is empty');
    }
  }

  List<Tensor>? getWeight(String name) {
    return this.weightMap[name];
  }

  void addTensorArray(TensorArray tensorArray) {
    this.tensorArrayMap[tensorArray.id] = tensorArray;
  }

  TensorArray? getTensorArray(int id) {
    return this.tensorArrayMap[id];
  }

  void addTensorList(TensorListContainer tensorList) {
    this.tensorListMap[tensorList.id] = tensorList;
  }

  TensorListContainer? getTensorList(int id) {
    return this.tensorListMap[id];
  }

  void dispose(Set<int> keepIds) {
    for (final array in this.tensorArrayMap.values) {
      array.clearAndClose(keepIds);
    }

    for (final array in this.tensorListMap.values) {
      array.clearAndClose(keepIds);
    }
  }
}
