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

// import {KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';
// import {Fill, FillAttrs} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import '_prelude.dart';

TensorInfo fill({
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
  required BackendWasm backend,
}) {
  final shape = attrs!['shape'] as List<int>;
  final value = attrs['value'] as Object;
  final dtype = attrs['dtype'] as DataType;

  final out = backend.makeOutput(shape, dtype);
  final outVals = backend.typedArrayFromHeap(out) as List;
  outVals.fillRange(0, outVals.length, value);
  return out;
}

final fillConfig = KernelConfigG(
  kernelName: Fill,
  backendName: 'wasm',
  kernelFunc: fill,
);
