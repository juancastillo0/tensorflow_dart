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

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/environment.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'util_base.dart' as util;

// import {ENGINE} from './engine';
// import {env} from './environment';
// import {Tensor} from './tensor';
// import {DataType, TensorLike} from './types';
// import {assert, flatten, inferDtype, isTypedArray, toTypedArray} from './util';

typedef TensorLike = Object;
// TypedArray|number|boolean|string|RecursiveArray<number|number[]|TypedArray>|
// RecursiveArray<boolean>|RecursiveArray<string>|Uint8Array[];

List<int> inferShape(TensorLike val, [DataType? dtype]) {
  var firstElem = val;

  if (val is TypedData) {
    return dtype == 'string' ? [] : [(val as List).length];
  }
  if (val is! List) {
    return []; // Scalar.
  }
  final shape = <int>[];

  while (firstElem is List || firstElem is TypedData && dtype != 'string') {
    shape.add((firstElem as List).length);
    firstElem = firstElem[0];
  }
  if (val is List && env().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')) {
    _deepAssertShapeConsistency(val, shape, []);
  }

  return shape;
}

_deepAssertShapeConsistency(
  TensorLike val,
  List<int> shape,
  List<int> indices,
) {
  // indices = indices || [];
  if (val is! List && val is! TypedData) {
    assert(
        shape.length == 0,
        () =>
            "Element arr[${indices.join('][')}] is a primitive, " +
            'but should be an array/TypedArray of ${shape[0]} elements');
    return;
  }
  val = val as List;
  final length = val.length;
  assert(
      shape.length > 0,
      () =>
          "Element arr[${indices.join('][')}] should be a primitive, " +
          'but is an array of ${length} elements');
  assert(
      length == shape[0],
      () =>
          "Element arr[${indices.join('][')}] should have ${shape[0]} " +
          'elements, but has ${length} elements');
  final subShape = shape.sublist(1);
  for (int i = 0; i < length; ++i) {
    _deepAssertShapeConsistency(val[i], subShape, [...indices, i]);
  }
}

void _assertDtype(
    // DataType|'numeric'|'string_or_numeric'
    String expectedDtype,
    DataType actualDType,
    String argName,
    String functionName) {
  if (expectedDtype == 'string_or_numeric') {
    return;
  }
  if (expectedDtype == null) {
    throw Exception('Expected dtype cannot be null.');
  }
  if (expectedDtype != 'numeric' && expectedDtype != actualDType ||
      expectedDtype == 'numeric' && actualDType == 'string') {
    throw Exception("Argument '${argName}' passed to '${functionName}' must " +
        'be ${expectedDtype} tensor, but got ${actualDType} tensor');
  }
}

T convertToTensor<T extends Tensor>(
  // TODO: TensorLike
  T x,
  String argName,
  String functionName,
  // DataType|'numeric'|'string_or_numeric'
  [
  String parseAsDtype = 'numeric',
]) {
  if (x is Tensor) {
    _assertDtype(parseAsDtype, x.dtype, argName, functionName);
    return x;
  }
  DataType inferredDtype = util.inferDtype(x);
  // If the user expects a bool/int/float, use that info to update the
  // inferredDtype when it is not a string.
  if (inferredDtype != 'string' &&
      ['bool', 'int32', 'float32'].indexOf(parseAsDtype) >= 0) {
    inferredDtype = parseAsDtype as DataType;
  }
  _assertDtype(parseAsDtype, inferredDtype, argName, functionName);

  if ((x == null) || (x is! List && x is! num && x is! bool && x is! String)) {
    final type = x == null ? 'null' : '${x.runtimeType}:${x}';
    throw Exception(
        "Argument '${argName}' passed to '${functionName}' must be a " +
            "Tensor or TensorLike, but got '${type}'");
  }
  final inferredShape = inferShape(x, inferredDtype);
  final List xList;
  if (x is! TypedData && x is! List) {
    xList = [x as num];
  } else {
    xList = x as List;
  }
  final values = inferredDtype != 'string'
      ? util.toTypedArray(xList, inferredDtype as DataType) as List
      : util.flatten(xList as List<String>, skipTypedArray: true);
  return ENGINE.makeTensor(values, inferredShape, inferredDtype) as T;
}

List<T> convertToTensorArray<T extends Tensor>(
  List<T> arg,
  String argName,
  String functionName,
  // DataType|'numeric'|'string_or_numeric'
  [
  String parseAsDtype = 'numeric',
]) {
  if (arg is! List) {
    throw Exception('Argument ${argName} passed to ${functionName} must be a ' +
        '`Tensor[]` or `TensorLike[]`');
  }
  final tensors = arg as List<T>;
  int i = 0;
  return tensors
      .map((t) => convertToTensor(
            t,
            '${argName}[${i++}]',
            functionName,
            parseAsDtype,
          ))
      .toList();
}
