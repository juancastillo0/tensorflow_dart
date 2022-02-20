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
// import {HashTableMap, NamedTensorMap} from '../data/types';
// import {HashTable} from './hash_table';

import 'package:tensorflow_wasm/src/converter/executor/hash_table.dart';
import 'package:tensorflow_wasm/src/converter/executor/tensor_array.dart';
import 'package:tensorflow_wasm/src/converter/executor/tensor_list.dart';
import 'package:tensorflow_wasm/src/tensor.dart' hide TensorList;

typedef TensorArrayMap = Map<int, TensorArray>;

typedef TensorListMap = Map<int, TensorListContainer>;

typedef HashTableMap = Map<int, HashTable>;

// class TensorInfo {
//   final String name;
//   final List<int>? shape;
//   final DataType dtype;

//   const TensorInfo({
//     required this.name,
//     this.shape,
//     required this.dtype,
//   });
// }

/**
 * Contains global resources of a model.
 */
class ResourceManager {
  final NamedTensorMap hashTableNameToHandle;
  final HashTableMap hashTableMap;

  ResourceManager([
    NamedTensorMap? hashTableNameToHandle,
    HashTableMap? hashTableMap,
  ])  : hashTableNameToHandle = hashTableNameToHandle ?? {},
        hashTableMap = hashTableMap ?? {};

  /**
   * Register a `HashTable` in the resource manager.
   *
   * The `HashTable` can be retrieved by `resourceManager.getHashTableById`,
   * where id is the table handle tensor's id.
   *
   * @param name Op node name that creates the `HashTable`.
   * @param hashTable The `HashTable` to be added to resource manager.
   */
  void addHashTable(String name, HashTable hashTable) {
    this.hashTableNameToHandle[name] = hashTable.handle;
    this.hashTableMap[hashTable.id] = hashTable;
  }

  /**
   * Get the table handle by node name.
   * @param name Op node name that creates the `HashTable`. This name is also
   *     used in the inputs list of lookup and import `HashTable` ops.
   */
  getHashTableHandleByName(String name) {
    return this.hashTableNameToHandle[name];
  }

  /**
   * Get the actual `HashTable` by its handle tensor's id.
   * @param id The id of the handle tensor.
   */
  HashTable? getHashTableById(int id) {
    return this.hashTableMap[id];
  }

  /**
   * Dispose `ResourceManager`, including its hashTables and tensors in them.
   */
  void dispose() {
    for (final key in this.hashTableMap.keys.toList()) {
      this.hashTableMap[key]!.clearAndClose();
      this.hashTableMap.remove(key);
    }

    for (final name in this.hashTableNameToHandle.keys.toList()) {
      this.hashTableNameToHandle[name]!.dispose();
      this.hashTableNameToHandle.remove(name);
    }
  }
}
