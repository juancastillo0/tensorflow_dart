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

// import {KernelConfig, KernelFunc, NonMaxSuppressionV4, NonMaxSuppressionV4Attrs, NonMaxSuppressionV4Inputs, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {parseResultStruct} from './NonMaxSuppression_util';

import '_prelude.dart';
import 'non_max_suppression_util.dart';

late final Function(List) _wasmFunc;
// : (
//     boxesId: number, scoresId: number, maxOutputSize: number,
//     iouThreshold: number, scoreThreshold: number,
//     padToMaxOutputSize: boolean) => number;

void _setup(BackendWasm backend) {
  _wasmFunc = backend.wasm.cwrap(NonMaxSuppressionV4, 'number', // Result*
      [
        'number', // boxesId
        'number', // scoresId
        'number', // maxOutputSize
        'number', // iouThreshold
        'number', // scoreThreshold
        'bool', // padToMaxOutputSize
      ]);
}

TensorInfoList nonMaxSuppressionV4({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final scores = inputs['scores']!;
  final boxes = inputs['boxes']!;

  final maxOutputSize = attrs!['maxOutputSize'] as int;
  final scoreThreshold = attrs['scoreThreshold'] as double;
  final iouThreshold = attrs['iouThreshold'] as double;
  final padToMaxOutputSize = attrs['padToMaxOutputSize'] as bool;

  final boxesId = backend.dataIdMap.get(boxes.dataId)!.id;
  final scoresId = backend.dataIdMap.get(scores.dataId)!.id;

  final resOffset = _wasmFunc([
    boxesId,
    scoresId,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
    padToMaxOutputSize
  ]);

  final result = parseResultStruct(backend, resOffset);

  // Since we are not using scores for V4, we have to delete it from the heap.
  backend.wasm.free(result.pSelectedScores);

  final selectedIndicesTensor = backend
      .makeOutput([result.selectedSize], 'int32', result.pSelectedIndices);

  final validOutputsTensor =
      backend.makeOutput([], 'int32', result.pValidOutputs);

  return TensorInfoList([selectedIndicesTensor, validOutputsTensor]);
}

final nonMaxSuppressionV4Config = KernelConfigG(
  kernelName: NonMaxSuppressionV4,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: nonMaxSuppressionV4,
);
