import 'package:tensorflow_wasm/tensorflow_wasm.dart';

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

// import {Tensor} from '../tensor';

// import * as broadcast_util from './broadcast_util';
// import {elu} from './elu';
// import {Activation} from './fused_types';
// import {leakyRelu} from './leaky_relu';
// import {mul} from './mul';
// import {prelu} from './prelu';
// import {relu} from './relu';
// import {relu6} from './relu6';
// import {reshape} from './reshape';
// import {sigmoid} from './sigmoid';
// import {step} from './step';
// import {sum} from './sum';

import 'broadcast_util.dart' as broadcast_util;
import 'fused_types.dart';

// Returns gradient for fused activation.
Tensor getFusedDyActivation(Tensor dy, Tensor y, Activation activation) {
  if (activation == null || activation == Activation.linear) {
    return dy;
  }
  if (activation == Activation.relu) {
    return mul(dy, step(y));
  }
  throw Exception(
      'Cannot compute gradient for fused activation ${activation}.');
}

// Returns gradient for fused bias.
Tensor getFusedBiasGradient(Tensor bias, Tensor dyActivation) {
  var res = dyActivation;
  final reduceAxes =
      broadcast_util.getReductionAxes(bias.shape, dyActivation.shape);
  if (reduceAxes.length > 0) {
    res = sum(res, reduceAxes);
  }
  return reshape(res, bias.shape);
}

Tensor applyActivation(
  Tensor x,
  Activation activation, {
  Tensor? preluActivationWeights,
  double? leakyreluAlpha,
}) {
  if (activation == Activation.linear) {
    return x;
  } else if (activation == Activation.relu) {
    return relu(x);
  } else if (activation == Activation.elu) {
    return elu(x);
  } else if (activation == Activation.relu6) {
    return relu6(x);
  } else if (activation == Activation.prelu) {
    return prelu(x, preluActivationWeights!);
  } else if (activation == Activation.leakyrelu) {
    return leakyRelu(x, leakyreluAlpha!);
  } else if (activation == Activation.sigmoid) {
    return sigmoid(x);
  }
  throw Exception('Unknown fused activation ${activation}.');
}

// Whether we should call fused ops.
bool shouldFuse(int gradientDepth, Activation activation) {
  final gradientMode = gradientDepth > 0;
  return !gradientMode || activation == Activation.linear;
}
