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

// import {ExpandDims, ExpandDimsAttrs, ExpandDimsInputs, KernelConfig, KernelFunc, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';
// import {reshape} from './Reshape';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

import 'reshape.dart';

TensorInfo expandDims({
  required NamedTensorInfoMap inputs,
  NamedAttrMap? attrs,
  required BackendWasm backend,
}) {
  final input = inputs['input']!;
  final dim = attrs!['dim'] as int;

  final inputRank = input.shape.length;
  final newShape = [...input.shape];
  int $dim = dim;
  if (dim < 0) {
    // Negative value is counted from the tail of rank.
    util.assert_(
        -(inputRank + 1) <= dim,
        () =>
            'Axis must be in the interval [${-(inputRank + 1)}, ${inputRank}]');
    $dim = inputRank + dim + 1;
  }
  newShape.insert($dim, 1);

  return reshape(
    inputs: {'x': input},
    backend: backend,
    attrs: {'shape': newShape},
  );
}

final expandDimsConfig = KernelConfigG<BackendWasm, NamedAttrMap>(
  kernelName: ExpandDims,
  backendName: 'wasm',
  kernelFunc: expandDims,
);
