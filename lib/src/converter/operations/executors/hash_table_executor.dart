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

// import {DataType, Tensor} from '@tensorflow/tfjs-core';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {HashTable} from '../../executor/hash_table';
// import {ResourceManager} from '../../executor/resource_manager';
// import {InternalOpAsyncExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import '../../executor/hash_table.dart';
import '../../executor/resource_manager.dart';
import '_prelude.dart';

Future<List<Tensor>> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
  ResourceManager resourceManager,
) async {
  switch (node.op) {
    case 'HashTable':
    case 'HashTableV2':
      {
        final keyDType =
            getParamValue('keyDType', node, tensorMap, context) as DataType;
        final valueDType =
            getParamValue('valueDType', node, tensorMap, context) as DataType;

        final hashTable = HashTable(keyDType, valueDType);
        resourceManager.addHashTable(node.name, hashTable);
        return [hashTable.handle];
      }
    case 'LookupTableImport':
    case 'LookupTableImportV2':
      {
        final handle = getParamValue(
            'tableHandle', node, tensorMap, context, resourceManager) as Tensor;
        final keys = getParamValue('keys', node, tensorMap, context) as Tensor;
        final values =
            getParamValue('values', node, tensorMap, context) as Tensor;

        final hashTable = resourceManager.getHashTableById(handle.id)!;

        return [await hashTable.import_(keys, values)];
      }
    case 'LookupTableFind':
    case 'LookupTableFindV2':
      {
        final handle = getParamValue(
            'tableHandle', node, tensorMap, context, resourceManager) as Tensor;
        final keys = getParamValue('keys', node, tensorMap, context) as Tensor;
        final defaultValue =
            getParamValue('defaultValue', node, tensorMap, context) as Tensor;

        final hashTable = resourceManager.getHashTableById(handle.id)!;
        return [await hashTable.find(keys, defaultValue)];
      }
    case 'LookupTableSize':
    case 'LookupTableSizeV2':
      {
        final handle = getParamValue(
            'tableHandle', node, tensorMap, context, resourceManager) as Tensor;

        final hashTable = resourceManager.getHashTableById(handle.id)!;
        return [hashTable.tensorSize()];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'hash_table';
