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

// import {DataType, scalar, Tensor} from '@tensorflow/tfjs-core';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {TensorArray} from '../../executor/tensor_array';
// import {fromTensor, reserve, scatter, split} from '../../executor/tensor_list';
// import {InternalOpAsyncExecutor, Node} from '../types';

// import {cloneTensor, getParamValue, getTensor} from './utils';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '../../executor/tensor_array.dart';
import '../../executor/tensor_list.dart';
import '_prelude.dart' hide split;

Future<List<Tensor?>?> executeOp(
    Node node, NamedTensorsMap tensorMap, ExecutionContext context) async {
  switch (node.op) {
    case 'If':
    case 'StatelessIf':
      {
        final thenFunc =
            getParamValue('thenBranch', node, tensorMap, context) as String;
        final elseFunc =
            getParamValue('elseBranch', node, tensorMap, context) as String;
        final cond = getParamValue('cond', node, tensorMap, context) as Tensor;
        final args =
            getParamValue('args', node, tensorMap, context) as List<Tensor>;
        final condValue = await cond.data();
        if (condValue[0]) {
          return context.functionMap[thenFunc]!.executeFunctionAsync(
              args, context.tensorArrayMap, context.tensorListMap);
        } else {
          return context.functionMap[elseFunc]!.executeFunctionAsync(
              args, context.tensorArrayMap, context.tensorListMap);
        }
      }
    case 'While':
    case 'StatelessWhile':
      {
        final bodyFunc =
            getParamValue('body', node, tensorMap, context) as String;
        final condFunc =
            getParamValue('cond', node, tensorMap, context) as String;
        final args =
            getParamValue('args', node, tensorMap, context) as List<Tensor>;

        // Calculate the condition of the loop
        final condResult = (await context.functionMap[condFunc]!
            .executeFunctionAsync(
                args, context.tensorArrayMap, context.tensorListMap));
        final argIds = args.map((tensor) => tensor.id).toList();
        List condValue = await condResult[0].data();
        // Dispose the intermediate tensors for condition function
        condResult.forEach((tensor) {
          if (!tensor.kept && argIds.indexOf(tensor.id) == -1) {
            tensor.dispose();
          }
        });

        List<Tensor> result = args;

        while (condValue[0]) {
          // Record the previous result for intermediate tensor tracking
          final origResult = result;
          // Execution the body of the loop
          result = await context.functionMap[bodyFunc]!.executeFunctionAsync(
              result, context.tensorArrayMap, context.tensorListMap);
          final resultIds = result.map((tensor) => tensor.id).toList();

          // Dispose the intermediate tensor for body function that is not global
          // kept, not input/output of the body function
          origResult.forEach((tensor) {
            if (!tensor.kept &&
                argIds.indexOf(tensor.id) == -1 &&
                resultIds.indexOf(tensor.id) == -1) {
              tensor.dispose();
            }
          });

          // Recalcuate the condition of the loop using the latest results.
          final condResult = (await context.functionMap[condFunc]!
              .executeFunctionAsync(
                  result, context.tensorArrayMap, context.tensorListMap));
          condValue = await condResult[0].data();
          // Dispose the intermediate tensors for condition function
          condResult.forEach((tensor) {
            if (!tensor.kept &&
                argIds.indexOf(tensor.id) == -1 &&
                resultIds.indexOf(tensor.id) == -1) {
              tensor.dispose();
            }
          });
        }
        return result;
      }
    case 'LoopCond':
      {
        final pred = getParamValue('pred', node, tensorMap, context) as Tensor;
        return [cloneTensor(pred)];
      }
    case 'Switch':
      {
        final pred = getParamValue('pred', node, tensorMap, context) as Tensor;
        Tensor data = getParamValue('data', node, tensorMap, context) as Tensor;
        if (!data.kept) {
          data = cloneTensor(data);
        }
        // Outputs nodes :0 => false, :1 => true
        return (await pred.data())[0] == 1 ? [null, data] : [data, null];
      }
    case 'Merge':
      {
        final inputName = node.inputNames.firstWhereOrNull(
            (name) => getTensor(name, tensorMap, context) != null);
        if (inputName != null) {
          final data = getTensor(inputName, tensorMap, context)!;
          return [cloneTensor(data)];
        }
        return null;
      }
    case 'Enter':
      {
        final frameId =
            getParamValue('frameName', node, tensorMap, context) as String;
        final data =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        context.enterFrame(frameId);
        return [cloneTensor(data)];
      }
    case 'Exit':
      {
        final data =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        context.exitFrame();
        return [cloneTensor(data)];
      }
    case 'NextIteration':
      {
        final data =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        context.nextIteration();
        return [cloneTensor(data)];
      }
    case 'TensorArrayV3':
      {
        final size = getParamValue('size', node, tensorMap, context) as int;
        final dtype =
            getParamValue('dtype', node, tensorMap, context) as DataType;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>?;
        final dynamicSize =
            getParamValue('dynamicSize', node, tensorMap, context) as bool;
        final clearAfterRead =
            getParamValue('clearAfterRead', node, tensorMap, context) as bool;
        final identicalElementShapes =
            getParamValue('identicalElementShapes', node, tensorMap, context)
                as bool;
        final name = getParamValue('name', node, tensorMap, context) as String;
        final tensorArray = TensorArray(name, dtype, size, elementShape,
            identicalElementShapes, dynamicSize, clearAfterRead);
        context.addTensorArray(tensorArray);
        return [tensorArray.idTensor, tfOps.scalar(1.0)];
      }
    case 'TensorArrayWriteV3':
      {
        final id =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final index = getParamValue('index', node, tensorMap, context) as int;
        final writeTensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final writeTensorArray = context.getTensorArray(id.id)!;
        writeTensorArray.write(index, writeTensor);
        return [writeTensorArray.idTensor];
      }
    case 'TensorArrayReadV3':
      {
        final readId =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final readIndex =
            getParamValue('index', node, tensorMap, context) as int;
        final readTensorArray = context.getTensorArray(readId.id)!;
        return [readTensorArray.read(readIndex)];
      }
    case 'TensorArrayGatherV3':
      {
        final gatherId =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final gatherIndices =
            getParamValue('indices', node, tensorMap, context) as List<int>;
        final gatherDtype =
            getParamValue('dtype', node, tensorMap, context) as DataType;
        final gatherTensorArray = context.getTensorArray(gatherId.id)!;
        return [gatherTensorArray.gather(gatherIndices, gatherDtype)];
      }
    case 'TensorArrayScatterV3':
      {
        final scatterId =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final scatterIndices =
            getParamValue('indices', node, tensorMap, context) as List<int>;
        final scatterTensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final scatterTensorArray = context.getTensorArray(scatterId.id)!;
        scatterTensorArray.scatter(scatterIndices, scatterTensor);
        return [scatterTensorArray.idTensor];
      }
    case 'TensorArrayConcatV3':
      {
        final concatId =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final concatTensorArray = context.getTensorArray(concatId.id)!;
        final concatDtype =
            getParamValue('dtype', node, tensorMap, context) as DataType;
        return [concatTensorArray.concat(concatDtype)];
      }
    case 'TensorArraySplitV3':
      {
        final splitId =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final splitTensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final lengths =
            getParamValue('lengths', node, tensorMap, context) as List<int>;
        final splitTensorArray = context.getTensorArray(splitId.id)!;
        splitTensorArray.split(lengths, splitTensor);
        return [splitTensorArray.idTensor];
      }
    case 'TensorArraySizeV3':
      {
        final sizeId =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final sizeTensorArray = context.getTensorArray(sizeId.id)!;
        return [tfOps.scalar(sizeTensorArray.size(), 'int32')];
      }
    case 'TensorArrayCloseV3':
      {
        final closeId =
            getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
        final closeTensorArray = context.getTensorArray(closeId.id)!;
        closeTensorArray.clearAndClose();
        return [closeTensorArray.idTensor];
      }
    case 'TensorListSetItem':
      {
        final idTensor =
            getParamValue('tensorListId', node, tensorMap, context) as Tensor;
        final index = getParamValue('index', node, tensorMap, context) as int;
        final writeTensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final tensorList = context.getTensorList(idTensor.id)!;
        tensorList.setItem(index, writeTensor);
        return [tensorList.idTensor];
      }
    case 'TensorListGetItem':
      {
        final idTensor =
            getParamValue('tensorListId', node, tensorMap, context) as Tensor;
        final readIndex =
            getParamValue('index', node, tensorMap, context) as int;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;

        final elementDType =
            getParamValue('elementDType', node, tensorMap, context) as DataType;
        final tensorList = context.getTensorList(idTensor.id)!;
        return [tensorList.getItem(readIndex, elementShape, elementDType)];
      }
    case 'TensorListScatterV2':
    case 'TensorListScatter':
      {
        final scatterIndices =
            getParamValue('indices', node, tensorMap, context) as List<int>;
        final scatterTensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        final numElements =
            getParamValue('numElements', node, tensorMap, context) as int;
        final tensorList =
            scatter(scatterTensor, scatterIndices, elementShape, numElements);
        context.addTensorList(tensorList);
        return [tensorList.idTensor];
      }
    case 'TensorListReserve':
    case 'EmptyTensorList':
      {
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        final elementDtype =
            getParamValue('elementDType', node, tensorMap, context) as DataType;
        final String numElementsParam;

        if (node.op == 'TensorListReserve') {
          numElementsParam = 'numElements';
        } else {
          numElementsParam = 'maxNumElements';
        }

        final numElements =
            getParamValue(numElementsParam, node, tensorMap, context) as int;

        final tensorList = reserve(elementShape, elementDtype, numElements);
        context.addTensorList(tensorList);
        return [tensorList.idTensor];
      }
    case 'TensorListGather':
      {
        final gatherId =
            getParamValue('tensorListId', node, tensorMap, context) as Tensor;
        final gatherIndices =
            getParamValue('indices', node, tensorMap, context) as List<int>;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        final elementDtype =
            getParamValue('elementDType', node, tensorMap, context) as DataType;
        final tensorList = context.getTensorList(gatherId.id)!;
        return [tensorList.gather(gatherIndices, elementDtype, elementShape)];
      }
    case 'TensorListStack':
      {
        final idTensor =
            getParamValue('tensorListId', node, tensorMap, context) as Tensor;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        final elementDtype =
            getParamValue('elementDType', node, tensorMap, context) as DataType;
        final numElements =
            getParamValue('numElements', node, tensorMap, context) as int;
        final tensorList = context.getTensorList(idTensor.id)!;
        return [tensorList.stack(elementShape, elementDtype, numElements)];
      }
    case 'TensorListFromTensor':
      {
        final tensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        final elementDtype =
            getParamValue('elementDType', node, tensorMap, context) as DataType;
        final tensorList = fromTensor(tensor, elementShape, elementDtype);
        context.addTensorList(tensorList);
        return [tensorList.idTensor];
      }
    case 'TensorListConcat':
      {
        final concatId =
            getParamValue('tensorListId', node, tensorMap, context) as Tensor;
        final tensorList = context.getTensorList(concatId.id)!;
        final concatDtype =
            getParamValue('dtype', node, tensorMap, context) as DataType;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        return [tensorList.concat(concatDtype, elementShape)];
      }
    case 'TensorListPushBack':
      {
        final idTensor =
            getParamValue('tensorListId', node, tensorMap, context) as Tensor;
        final writeTensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final tensorList = context.getTensorList(idTensor.id)!;
        tensorList.pushBack(writeTensor);
        return [tensorList.idTensor];
      }
    case 'TensorListPopBack':
      {
        final idTensor =
            getParamValue('tensorListId', node, tensorMap, context) as Tensor;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        final elementDType =
            getParamValue('elementDType', node, tensorMap, context) as DataType;
        final tensorList = context.getTensorList(idTensor.id)!;
        return [tensorList.popBack(elementShape, elementDType)];
      }
    case 'TensorListSplit':
      {
        final splitTensor =
            getParamValue('tensor', node, tensorMap, context) as Tensor;
        final elementShape =
            getParamValue('elementShape', node, tensorMap, context)
                as List<int>;
        final lengths =
            getParamValue('lengths', node, tensorMap, context) as List<int>;

        final tensorList = split(splitTensor, lengths, elementShape);
        context.addTensorList(tensorList);
        return [tensorList.idTensor];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'control';
