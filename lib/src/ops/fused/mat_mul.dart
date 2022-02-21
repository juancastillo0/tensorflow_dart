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

// import {ENGINE} from '../../engine';
// import {customGrad} from '../../gradients';
// import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs} from '../../kernel_names';
// import {NamedAttrMap} from '../../kernel_registry';
// import {Tensor, Tensor3D} from '../../tensor';
// import {GradSaveFunc, NamedTensorMap} from '../../tensor_types';
// import {makeTypesMatch} from '../../tensor_util';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import * as util from '../../util';

// import {add} from '../add';
// import * as broadcast_util from '../broadcast_util';
// import {Activation} from '../fused_types';
// import {applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse} from '../fused_util';
// import {matMul as unfusedMatMul} from '../mat_mul';
// import {op} from '../operation';
// import {reshape} from '../reshape';

import 'package:tensorflow_wasm/src/gradients.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

import '../_prelude.dart';
import '../broadcast_util.dart' as broadcast_util;
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import '../fused_types.dart';
import '../fused_util.dart';
import '../mat_mul.dart' as __mat_mul;

const unfusedMatMul = __mat_mul.matMul;

/**
 * Computes the dot product of two matrices with optional activation and bias.
 *
 * ```js
 * const a = tf.tensor2d([-1, -2], [1, 2]);
 * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const bias = tf.tensor2d([1, 2], [1, 2]);
 *
 * tf.fused.matMul({a, b, bias, activation: 'relu'}).print();
 * ```
 *
 * @param obj An object with the following properties:
 * - `a` First matrix in dot product operation.
 * - `b` Second matrix in dot product operation.
 * - `transposeA` If true, `a` is transposed before multiplication.
 * - `transposeB` If true, `b` is transposed before multiplication.
 * - `bias` Matrix to be added to the result.
 * - `activation` Name of activation kernel (defaults to `linear`).
 * - `preluActivationWeights` Tensor of prelu weights.
 * - `leakyreluAlpha` Alpha of leakyrelu.
 */
Tensor fusedMatMul({
  required Tensor a,
  required Tensor b,
  bool transposeA = false,
  bool transposeB = false,
  Tensor? bias,
  Activation activation = Activation.linear,
  Tensor? preluActivationWeights,
  double? leakyreluAlpha,
}) {
  if (shouldFuse(ENGINE.state.gradientDepth, activation) == false) {
    var result =
        unfusedMatMul(a, b, transposeA: transposeA, transposeB: transposeB);
    if (bias != null) {
      result = add(result, bias);
    }

    return applyActivation(result, activation,
        preluActivationWeights: preluActivationWeights,
        leakyreluAlpha: leakyreluAlpha);
  }

  var $a = convertToTensor(a, 'a', 'fused matMul');
  var $b = convertToTensor(b, 'b', 'fused matMul');
  final _f = makeTypesMatch($a, $b);
  $a = _f.first;
  $b = _f.second;

  final innerShapeA =
      transposeA ? $a.shape[$a.rank - 2] : $a.shape[$a.rank - 1];
  final innerShapeB =
      transposeB ? $b.shape[$b.rank - 1] : $b.shape[$b.rank - 2];

  final outerShapeA =
      transposeA ? $a.shape[$a.rank - 1] : $a.shape[$a.rank - 2];
  final outerShapeB =
      transposeB ? $b.shape[$b.rank - 2] : $b.shape[$b.rank - 1];

  final outerDimsA = $a.shape.slice(0, -2);
  final outerDimsB = $b.shape.slice(0, -2);
  final batchDimA = util.sizeFromShape(outerDimsA);
  final batchDimB = util.sizeFromShape(outerDimsB);

  util.assert_(
      innerShapeA == innerShapeB,
      () =>
          'Error in fused matMul: inner shapes (${innerShapeA}) and (' +
          '${innerShapeB}) of Tensors with shapes ${$a.shape} and ' +
          '${$b.shape} and transposeA=${transposeA}' +
          ' and transposeB=${transposeB} must match.');

  final outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(
      $a.shape.slice(0, -2), $b.shape.slice(0, -2));
  final outShape = [...outShapeOuterDims, outerShapeA, outerShapeB];

  final Tensor3D a3D = transposeA
      ? reshape($a, [batchDimA, innerShapeA, outerShapeA])
      : reshape($a, [batchDimA, outerShapeA, innerShapeA]);
  final Tensor3D b3D = transposeB
      ? reshape($b, [batchDimB, outerShapeB, innerShapeB])
      : reshape($b, [batchDimB, innerShapeB, outerShapeB]);

  Tensor? $bias;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused matMul');
    $bias = makeTypesMatch($bias, $a).first;

    broadcast_util.assertAndGetBroadcastShape(outShape, $bias.shape);
  }

  Tensor? $preluActivationWeights;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused matMul');
  }

  grad(Tensor3D dy, List<Tensor> saved) {
    final a3D = saved[0];
    final b3D = saved[0];
    final y = saved[0];
    final $bias = saved[0];
    // we reshape dy because the result of the forward is not
    // necessarily going to be a 3d tensor due to a reshape done at the end of
    // the customOp.
    final dyActivation =
        getFusedDyActivation(reshape(dy, y.shape), y, activation);
    final Tensor aDer;
    final Tensor bDer;

    if (!transposeA && !transposeB) {
      aDer =
          unfusedMatMul(dyActivation, b3D, transposeA: false, transposeB: true);
      bDer =
          unfusedMatMul(a3D, dyActivation, transposeA: true, transposeB: false);
    } else if (!transposeA && transposeB) {
      aDer = unfusedMatMul(dyActivation, b3D,
          transposeA: false, transposeB: false);
      bDer =
          unfusedMatMul(dyActivation, a3D, transposeA: true, transposeB: false);
    } else if (transposeA && !transposeB) {
      aDer =
          unfusedMatMul(b3D, dyActivation, transposeA: false, transposeB: true);
      bDer = unfusedMatMul(a3D, dyActivation,
          transposeA: false, transposeB: false);
    } else {
      aDer =
          unfusedMatMul(b3D, dyActivation, transposeA: true, transposeB: true);
      bDer =
          unfusedMatMul(dyActivation, a3D, transposeA: true, transposeB: true);
    }

    if ($bias != null) {
      final biasDer = getFusedBiasGradient($bias, dyActivation);
      return TensorList([aDer, bDer, biasDer]);
    } else {
      return TensorList([aDer, bDer]);
    }
  }

  ;

  final inputs = {
    // : _FusedMatMulInputs
    'a': a3D,
    'b': b3D,
    if ($bias != null) 'bias': $bias,
    if ($preluActivationWeights != null)
      'preluActivationWeights': $preluActivationWeights
  };
  final attrs = // : _FusedMatMulAttrs
      {
    'transposeA': transposeA,
    'transposeB': transposeB,
    'activation': activation,
    'leakyreluAlpha': leakyreluAlpha,
  };

  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if ($bias == null) {
    final customOp = customGrad((tensorInputs, save) {
      final res =
          // tslint:disable-next-line: no-unnecessary-type-assertion
          ENGINE.runKernel(FusedMatMul_, inputs, attrs) as Tensor;

      save([...tensorInputs, res]);

      return Gradient(reshape(res, outShape), grad);
    });
    return customOp([a3D, b3D]);
  } else {
    final customOpWithBias = customGrad((tensorInputs, save) {
      final res =
          // tslint:disable-next-line: no-unnecessary-type-assertion
          ENGINE.runKernel(FusedMatMul_, inputs, attrs) as Tensor;

      save([...tensorInputs, res]);

      return Gradient(reshape(res, outShape), grad);
    });

    return customOpWithBias([a3D, b3D, $bias]);
  }
}
