/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
// import {DataType, keep, scalar, stack, Tensor, tidy, unstack, util} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

/**
 * Hashtable contains a set of tensors, which can be accessed by key.
 */
class HashTable {
  final Tensor handle;

  // tslint:disable-next-line: no-any
  final Map<dynamic, Tensor> _tensorMap = {};

  get id {
    return this.handle.id;
  }

  final DataType keyDType;
  final DataType valueDType;

  /**
   * Constructor of HashTable. Creates a hash table.
   *
   * @param keyDType `dtype` of the table keys.
   * @param valueDType `dtype` of the table values.
   */
  HashTable(this.keyDType, this.valueDType) : handle = scalar(0) {
    keep(this.handle);
  }

  /**
   * Dispose the tensors and handle and clear the hashtable.
   */
  clearAndClose() {
    this._tensorMap.values.forEach((value) => value.dispose());
    this._tensorMap.clear();
    this.handle.dispose();
  }

  /**
   * The number of items in the hash table.
   */
  int size() {
    return this._tensorMap.length;
  }

  /**
   * The number of items in the hash table as a rank-0 tensor.
   */
  Tensor tensorSize() {
    // TODO: was `tfOps.scalar`
    return scalar(this.size(), 'int32');
  }

  /**
   * Replaces the contents of the table with the specified keys and values.
   * @param keys Keys to store in the hashtable.
   * @param values Values to store in the hashtable.
   */
  Future<Tensor> import_(Tensor keys, Tensor values) async {
    this._checkKeyAndValueTensor(keys, values);

    // We only store the primitive values of the keys, this allows lookup
    // to be O(1).
    final $keys = await keys.data();

    // Clear the hashTable before inserting new values.
    this._tensorMap.values.forEach((value) => value.dispose());
    this._tensorMap.clear();

    return tidy(() {
      final $values = unstack(values);

      final keysLength = $keys.length;
      final valuesLength = $values.length;

      util.assert_(
          keysLength == valuesLength,
          () =>
              "The number of elements doesn't match, keys has " +
              "${keysLength} elements, the values has ${valuesLength} " +
              "elements.");

      for (int i = 0; i < keysLength; i++) {
        final key = $keys[i];
        final value = $values[i];

        keep(value);
        this._tensorMap.set(key, value);
      }

      return this.handle;
    });
  }

  /**
   * Looks up keys in a hash table, outputs the corresponding values.
   *
   * Performs batch lookups, for every element in the key tensor, !find`
   * stacks the corresponding value into the return tensor.
   *
   * If an element is not present in the table, the given `defaultValue` is
   * used.
   *
   * @param keys Keys to look up. Must have the same type as the keys of the
   *     table.
   * @param defaultValue The scalar `defaultValue` is the value output for keys
   *     not present in the table. It must also be of the same type as the
   *     table values.
   */
  Future<Tensor> find(Tensor keys, Tensor defaultValue) async {
    this._checkKeyAndValueTensor(keys, defaultValue);

    final $keys = await keys.data();

    return tidy(() {
      final List<Tensor> result = [];

      for (int i = 0; i < $keys.length; i++) {
        final key = $keys[i];

        final value = this._findWithDefault(key, defaultValue);
        result.add(value);
      }

      return stack(result);
    });
  }

  // tslint:disable-next-line: no-any
  Tensor _findWithDefault(key, Tensor defaultValue) {
    final result = this._tensorMap.get(key);

    return result != null ? result : defaultValue;
  }

  _checkKeyAndValueTensor(Tensor key, Tensor value) {
    if (key.dtype != this.keyDType) {
      throw Exception(
          'Expect key dtype ${this.keyDType}, but got ' + '${key.dtype}');
    }

    if (value.dtype != this.valueDType) {
      throw Exception(
          'Expect value dtype ${this.valueDType}, but got ' + '${value.dtype}');
    }
  }
}
