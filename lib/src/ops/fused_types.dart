/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

// import {Tensor, Tensor3D, Tensor4D} from '../tensor';
// import {Conv2DInfo} from './conv_util';

// export type FusedConv2DConfig = {
//   input: Tensor4D,
//   filter: Tensor4D,
//   convInfo: Conv2DInfo,
//   bias?: Tensor,
//   activation?: Activation,
//   preluActivationWeights?: Tensor,
//   leakyreluAlpha?: number
// };

// export type FusedBatchMatMulConfig = {
//   a: Tensor3D,
//   b: Tensor3D,
//   transposeA: boolean,
//   transposeB: boolean,
//   bias?: Tensor,
//   activation?: Activation,
//   preluActivationWeights?: Tensor,
//   leakyreluAlpha?: number
// };

enum Activation {
  linear,
  relu,
  prelu,
  elu,
  relu6,
  leakyrelu,
  sigmoid,
}
