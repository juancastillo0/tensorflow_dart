/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

// import {KernelConfig, KernelFunc, SparseSegmentMean, SparseSegmentMeanInputs, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {setup, sparseSegmentReduction} from './SparseSegmentReduction';

import '_prelude.dart';
import 'sparse_segment_reduction.dart';

TensorInfo sparseSegmentMean({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  return sparseSegmentReduction(backend: backend, inputs: inputs, isMean: true);
}

final sparseSegmentMeanConfig = KernelConfigG(
  kernelName: SparseSegmentMean,
  backendName: 'wasm',
  setupFunc: setupSparseSegmentReduction,
  kernelFunc: sparseSegmentMean,
);
