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
/**
 * This differs from util.assertShapesMatch in that it allows values of
 * negative one, an undefined size of a dimensinon, in a shape to match
 * anything.
 */

// import {Tensor, util} from '@tensorflow/tfjs-core';

import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

/**
 * Used by TensorList and TensorArray to verify if elementShape matches, support
 * negative value as the dim shape.
 * @param shapeA
 * @param shapeB
 * @param errorMessagePrefix
 */
void assertShapesMatchAllowUndefinedSize(
  // number|number[]
  List<int> shapeA,
  // number|number[]
  List<int> shapeB, [
  String errorMessagePrefix = '',
]) {
  // constant shape means unknown rank
  if (shapeA is int || shapeB is int) {
    return;
  }
  util.assert_(shapeA.length == shapeB.length,
      () => errorMessagePrefix + " Shapes ${shapeA} and ${shapeB} must match");
  for (int i = 0; i < shapeA.length; i++) {
    final dim0 = shapeA[i];
    final dim1 = shapeB[i];
    util.assert_(
        dim0 < 0 || dim1 < 0 || dim0 == dim1,
        () =>
            errorMessagePrefix + " Shapes ${shapeA} and ${shapeB} must match");
  }
}

bool fullDefinedShape(
  // number|number[]
  List<int> elementShape,
) {
  if (elementShape is int || elementShape.any((dim) => dim < 0)) {
    return false;
  }
  return true;
}

/**
 * Generate the output element shape from the list elementShape, list tensors
 * and input param.
 * @param listElementShape
 * @param tensors
 * @param elementShape
 */
List<int> inferElementShape(
  // number|number[]
  List<int> listElementShape,
  List<Tensor> tensors,
  // number|number[]
  List<int> elementShape,
) {
  var partialShape = mergeElementShape(listElementShape, elementShape);
  final notfullDefinedShape = !fullDefinedShape(partialShape);
  if (notfullDefinedShape && tensors.length == 0) {
    throw Exception("Tried to calculate elements of an empty list" +
        " with non-fully-defined elementShape: ${partialShape}");
  }
  if (notfullDefinedShape) {
    tensors.forEach((tensor) {
      partialShape = mergeElementShape(tensor.shape, partialShape);
    });
  }
  if (!fullDefinedShape(partialShape)) {
    throw Exception("Non-fully-defined elementShape: ${partialShape}");
  }
  return partialShape;
}

List<int> mergeElementShape(
  // : number|number[]
  List<int> elementShapeA,
  // : number|number[]
  List<int> elementShapeB,
) {
  if (elementShapeA is int) {
    return elementShapeB;
  }
  if (elementShapeB is int) {
    return elementShapeA;
  }

  if (elementShapeA.length != elementShapeB.length) {
    throw Exception(
        "Incompatible ranks during merge: ${elementShapeA} vs. ${elementShapeB}");
  }

  final List<int> result = [];
  for (int i = 0; i < elementShapeA.length; ++i) {
    final dim0 = elementShapeA[i];
    final dim1 = elementShapeB[i];
    if (dim0 >= 0 && dim1 >= 0 && dim0 != dim1) {
      throw Exception(
          "Incompatible shape during merge: ${elementShapeA} vs. ${elementShapeB}");
    }
    result[i] = dim0 >= 0 ? dim0 : dim1;
  }
  return result;
}
