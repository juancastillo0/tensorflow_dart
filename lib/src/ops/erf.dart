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

import {ENGINE} from '../engine';
import {Erf, ErfInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {cast} from './cast';
import {op} from './operation';

export const ERF_P = 0.3275911;
export const ERF_A1 = 0.254829592;
export const ERF_A2 = -0.284496736;
export const ERF_A3 = 1.421413741;
export const ERF_A4 = -1.453152027;
export const ERF_A5 = 1.061405429;

/**
 * Computes gause error function of the input `tf.Tensor` element-wise:
 * `erf(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, .1, -.1, .7]);
 *
 * x.erf().print(); // or tf.erf(x);
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function erf_<T extends Tensor>(x: T|TensorLike): T {
  let $x = convertToTensor(x, 'x', 'erf');
  util.assert(
      $x.dtype === 'int32' || $x.dtype === 'float32',
      () => 'Input dtype must be `int32` or `float32`.');

  if ($x.dtype === 'int32') {
    $x = cast($x, 'float32');
  }

  const inputs: ErfInputs = {x: $x};
  return ENGINE.runKernel(Erf, inputs as {} as NamedTensorMap);
}
export const erf = op({erf_});