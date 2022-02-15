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

import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/types.dart';
import 'package:tensorflow_wasm/src/util_base.dart';

// import {Tensor} from './tensor';
// import {TensorContainer, TensorContainerArray} from './tensor_types';
// import {upcastType} from './types';
// import {assert} from './util';

class Tuple<F, S> {
  final F first;
  final S second;

  Tuple(this.first, this.second);
}

Tuple<T, T> makeTypesMatch<T extends Tensor>(T a, T b) {
  if (a.dtype == b.dtype) {
    return Tuple(a, b);
  }
  final dtype = upcastType(a.dtype, b.dtype);
  return Tuple(a.cast(dtype), b.cast(dtype));
}

void assertTypesMatch(Tensor a, Tensor b) {
  assert_(
      a.dtype == b.dtype,
      () =>
          'The dtypes of the first(${a.dtype}) and' +
          ' second(${b.dtype}) input must match');
}

bool isTensorInList(Tensor tensor, List<Tensor> tensorList) {
  return tensorList.any((x) => x.id == tensor.id);
}

/**
 * Extracts any `Tensor`s found within the provided object.
 *
 * @param container an object that may be a `Tensor` or may directly contain
 *   `Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. In general it
 *   is safe to pass any object here, except that `Promise`s are not
 *   supported.
 * @returns An array of `Tensors` found within the passed object. If the
 *   argument is simply a `Tensor', a list containing that `Tensor` is
 *   returned. If the object is not a `Tensor` or does not
 *   contain `Tensors`, an empty list is returned.
 */
List<Tensor> getTensorsInContainer(TensorContainer result) {
  final list = <Tensor>[];
  final seen = <Map?>{};
  _walkTensorContainer(result, list, seen);
  return list;
}

void _walkTensorContainer(
  TensorContainer container,
  List<Tensor> list,
  Set<Map<dynamic, dynamic>?> seen,
) {
  if (container == null) {
    return;
  }
  if (container is Tensor) {
    list.add(container);
    return;
  }
  if (container is! Iterable && container is! Map) {
    return;
  }
  // Iteration over keys works also for arrays.
  final iterable = container is Map ? container.values : container as Iterable;
  for (final val in iterable) {
    // final val = iterable[k];
    if (!seen.contains(val)) {
      seen.add(val);
      _walkTensorContainer(val, list, seen);
    }
  }
}

// // tslint:disable-next-line:no-any
// function isIterable(obj: any): boolean {
//   return Array.isArray(obj) || typeof obj === 'object';
// }