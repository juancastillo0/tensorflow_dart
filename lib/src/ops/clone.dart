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

// import {ENGINE} from '../engine';
// import {Identity, IdentityInputs} from '../kernel_names';
// import {Tensor} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {op} from './operation';

import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/kernel_names.dart';
import 'package:tensorflow_wasm/src/ops/operation.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/tensor_util_env.dart';

/**
 * Creates a new tensor with the same values and shape as the specified
 * tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 *
 * x.clone().print();
 * ```
 *
 * @param x The tensor to clone.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
T clone<T extends Tensor>(T x) {
  // TODO: | TensorLike
  return execOp('clone', () {
    final $x = convertToTensor(x, 'x', 'clone', 'string_or_numeric');
    final inputs = {'x': $x};

    // Note this op is called tf.identity in python. Hence the kernel name used
    // here.
    return ENGINE.runKernel(Identity, inputs) as T;
  });
}

// final clone = op({'clone_': _clone});
