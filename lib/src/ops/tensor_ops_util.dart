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

// import {ENGINE} from '../engine';
// import {Tensor} from '../tensor';
// import {TensorLike, TypedArray} from '../types';
// import {DataType} from '../types';
// import {assert, assertNonNegativeIntegerDimensions, flatten, inferDtype, isTypedArray, sizeFromShape, toTypedArray} from '../util';

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/tensor_util_env.dart';
import 'package:tensorflow_wasm/src/util_base.dart';

/** This is shared code across all tensor creation methods. */
Tensor makeTensor(
  TensorLike values,
  List<int>? shape,
  List<int> inferredShape, [
  DataType? dtype,
]) {
  dtype ??= inferDtype(values);

  if (dtype == 'complex64') {
    throw Exception('Cannot construct a complex64 tensor directly. ' +
        'Please use tf.complex(real, imag).');
  }
  if (values is! TypedData &&
      values is! List &&
      values is! num &&
      values is! bool &&
      values is! String) {
    throw Exception(
        'values passed to tensor(values) must be a number/boolean/string or ' +
            'an array of numbers/booleans/strings, or a TypedArray');
  }
  if (shape != null) {
    assertNonNegativeIntegerDimensions(shape);

    final providedSize = sizeFromShape(shape);
    final inferredSize = sizeFromShape(inferredShape);
    assert_(
        providedSize == inferredSize,
        () =>
            'Based on the provided shape, [${shape}], the tensor should have ' +
            '${providedSize} values but has ${inferredSize}');

    for (int i = 0; i < inferredShape.length; ++i) {
      final inferred = inferredShape[i];
      final flatDimsDontMatch = i == inferredShape.length - 1
          ? inferred != sizeFromShape(shape.sublist(i))
          : true;
      assert(
          inferredShape[i] == shape[i] || !flatDimsDontMatch,
          () =>
              'Error creating a new Tensor. Inferred shape ' +
              '(${inferredShape}) does not match the provided ' +
              'shape (${shape}). ');
    }
  }

  if (values is! TypedData && values is! List) {
    values = [values];
  }

  shape = shape ?? inferredShape;
  values = dtype != 'string'
      ? toTypedArray(values, dtype)
      : flatten(values as List<String>, skipTypedArray: true) as List<String>;
  return ENGINE.makeTensor(values as List, shape, dtype);
}
