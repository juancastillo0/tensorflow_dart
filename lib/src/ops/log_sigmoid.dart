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

// import {customGrad} from '../gradients';
// import {Tensor} from '../tensor';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike} from '../types';

// import {mul} from './mul';
// import {neg} from './neg';
// import {op} from './operation';
// import {sigmoid} from './sigmoid';
// import {softplus} from './softplus';

import '../gradients.dart';
import '_prelude.dart';
import 'ops.dart';

/**
 * Computes log sigmoid of the input `tf.Tensor` element-wise:
 * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.logSigmoid().print();  // or tf.logSigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
T logSigmoid<T extends Tensor>(T x) {
  return execOp('logSigmoid', () {
    final $x = convertToTensor(x, 'x', 'logSigmoid');

    // Use a custom gradient to maintain previous implementation.
    // There is no LogSigmoid kernel in TF so we can't use engine.runKernel
    // directly
    final customOp = customGrad((tensorInputs, _) {
      final x = tensorInputs.first;
      // TODO(yassogba) we can remove the chained softplus call here only
      // after backends have modualrized softplus at which point we can call
      // engine runKernel(..., Sotfplus, ...) directly.
      final value = neg(softplus(neg(x))) as T;

      gradFunc(T dy, _) {
        final derX = mul(dy, sigmoid(neg(x)));
        return derX;
      }

      return Gradient<T>(value, gradFunc);
    });

    return customOp([$x]);
  });
}
