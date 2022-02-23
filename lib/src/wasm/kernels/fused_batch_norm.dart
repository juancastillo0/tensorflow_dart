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

// import {FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';

import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmBatchNorm;
// : (    xId: number, meanId: number, varianceId: number, offsetId: number,
//     scaleId: number, varianceEpsilon: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmBatchNorm = backend.wasm.cwrap(FusedBatchNorm, null /* void */,
      ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
}

TensorInfo fusedBatchNorm({
  required BackendWasm backend,
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
}) {
  final varianceEpsilon = attrs!['varianceEpsilon'] as num;
  final x = inputs['x']!;
  final mean = inputs['mean']!;
  final variance = inputs['variance']!;
  final offset = inputs['offset'];
  final scale = inputs['scale'];
  final xId = backend.dataIdMap.get(x.dataId)!.id;
  final meanId = backend.dataIdMap.get(mean.dataId)!.id;
  final varianceId = backend.dataIdMap.get(variance.dataId)!.id;
  final offsetId =
      offset != null ? backend.dataIdMap.get(offset.dataId)!.id : 0;
  final scaleId = scale != null ? backend.dataIdMap.get(scale.dataId)!.id : 0;

  final out = backend.makeOutput(x.shape, x.dtype);
  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(x.shape) == 0) {
    return out;
  }

  final outId = backend.dataIdMap.get(out.dataId)!.id;

  _wasmBatchNorm(
      [xId, meanId, varianceId, offsetId, scaleId, varianceEpsilon, outId]);
  return out;
}

final fusedBatchNormConfig = KernelConfigG(
  kernelName: FusedBatchNorm,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: fusedBatchNorm,
);
