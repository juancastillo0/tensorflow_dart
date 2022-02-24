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

// import {KernelConfig, KernelFunc, OnesLike, OnesLikeInputs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';

TensorInfo onesLike({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final x = inputs['x']!;
  final out = backend.makeOutput(x.shape, x.dtype);
  final outVals = backend.typedArrayFromHeap(out) as List;
  outVals.fillRange(0, outVals.length, 1);
  return out;
}

final onesLikeConfig = KernelConfigG(
  kernelName: OnesLike,
  backendName: 'wasm',
  kernelFunc: onesLike,
);
