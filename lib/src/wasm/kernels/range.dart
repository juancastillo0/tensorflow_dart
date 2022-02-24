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

// import {KernelConfig, KernelFunc, Range, RangeAttrs, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {rangeImplCPU} from '../kernel_utils/shared';

import '../kernel_utils/shared.dart' show rangeImplCPU;
import '_prelude.dart';

TensorInfo range({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final start = attrs!['start'] as int;
  final stop = attrs['stop'] as int;
  final step = attrs['step'] as int;
  final dtype = attrs['dtype'] as DataType;
  final values = rangeImplCPU(start, stop, step, dtype);

  final out = backend.makeOutput([values.length], dtype);
  final outVals = backend.typedArrayFromHeap(out) as List;
  outVals.setAll(0, values);
  return out;
}

final rangeConfig = KernelConfigG(
  kernelName: Range,
  backendName: 'wasm',
  kernelFunc: range,
);
