// /**
//  * @license
//  * Copyright 2020 Google LLC. All Rights Reserved.
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  * =============================================================================
//  */

import 'dart:convert' show utf8;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:logging/logging.dart';

import 'package:tensorflow_wasm/src/environment.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/tensor_util_env.dart';

final log = Logger('tf_dart');

// import {DataType, DataTypeMap, FlatVector, NumericDataType, RecursiveArray, TensorLike, TypedArray} from './types';

// /**
//  * Shuffles the array in-place using Fisher-Yates algorithm.
//  *
//  * ```js
//  * const a = [1, 2, 3, 4, 5];
//  * tf.util.shuffle(a);
//  * console.log(a);
//  * ```
//  *
//  * @param array The array to shuffle in-place.
//  *
//  * @doc {heading: 'Util', namespace: 'util'}
//  */
// // tslint:disable-next-line:no-any
// export function shuffle(array: any[]|Uint32Array|Int32Array|
//                         Float32Array): void {
//   let counter = array.length;
//   let index = 0;
//   // While there are elements in the array
//   while (counter > 0) {
//     // Pick a random index
//     index = (Math.random() * counter) | 0;
//     // Decrease counter by 1
//     counter--;
//     // And swap the last element with it
//     swap(array, counter, index);
//   }
// }

// /**
//  * Shuffles two arrays in-place the same way using Fisher-Yates algorithm.
//  *
//  * ```js
//  * const a = [1,2,3,4,5];
//  * const b = [11,22,33,44,55];
//  * tf.util.shuffleCombo(a, b);
//  * console.log(a, b);
//  * ```
//  *
//  * @param array The first array to shuffle in-place.
//  * @param array2 The second array to shuffle in-place with the same permutation
//  *     as the first array.
//  *
//  * @doc {heading: 'Util', namespace: 'util'}
//  */
// export function shuffleCombo(
//     // tslint:disable-next-line:no-any
//     array: any[]|Uint32Array|Int32Array|Float32Array,
//     // tslint:disable-next-line:no-any
//     array2: any[]|Uint32Array|Int32Array|Float32Array): void {
//   if (array.length !== array2.length) {
//     throw new Error(
//         `Array sizes must match to be shuffled together ` +
//         `First array length was ${array.length}` +
//         `Second array length was ${array2.length}`);
//   }
//   let counter = array.length;
//   let index = 0;
//   // While there are elements in the array
//   while (counter > 0) {
//     // Pick a random index
//     index = (Math.random() * counter) | 0;
//     // Decrease counter by 1
//     counter--;
//     // And swap the last element of each array with it
//     swap(array, counter, index);
//     swap(array2, counter, index);
//   }
// }

/** Clamps a value to a specified range. */
int clamp(int min, int x, int max) {
  return math.max(min, math.min(x, max));
}

int nearestLargerEven(int val) {
  return val % 2 == 0 ? val : val + 1;
}

// export function swap<T>(
//     object: {[index: number]: T}, left: number, right: number) {
//   const temp = object[left];
//   object[left] = object[right];
//   object[right] = temp;
// }

int sumList(List<int> arr) {
  int sum = 0;
  for (int i = 0; i < arr.length; i++) {
    sum += arr[i];
  }
  return sum;
}

// /**
//  * Returns a sample from a uniform [a, b) distribution.
//  *
//  * @param a The minimum support (inclusive).
//  * @param b The maximum support (exclusive).
//  * @return A pseudorandom number on the half-open interval [a,b).
//  */
// export function randUniform(a: number, b: number) {
//   const r = Math.random();
//   return (b * r) + (1 - r) * a;
// }

// /** Returns the squared Euclidean distance between two vectors. */
// export function distSquared(a: FlatVector, b: FlatVector): number {
//   let result = 0;
//   for (let i = 0; i < a.length; i++) {
//     const diff = Number(a[i]) - Number(b[i]);
//     result += diff * diff;
//   }
//   return result;
// }

/**
 * Asserts that the expression is true. Otherwise throws an error with the
 * provided message.
 *
 * ```js
 * const x = 2;
 * tf.util.assert(x === 2, 'x is not 2');
 * ```
 *
 * @param expr The expression to assert (as a boolean).
 * @param msg A function that returns the message to report when throwing an
 *     error. We use a function for performance reasons.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
void assert_(bool expr, Object msg) {
  if (!expr) {
    throw Exception(msg is String ? msg : (msg as Function)());
  }
}

void assertShapesMatch(
  List<int> shapeA,
  List<int> shapeB, [
  String errorMessagePrefix = '',
]) {
  assert(arraysEqual(shapeA, shapeB),
      () => errorMessagePrefix + ' Shapes ${shapeA} and ${shapeB} must match');
}

// export function assertNonNull(a: TensorLike): void {
//   assert(
//       a != null,
//       () => `The input to the tensor constructor must be a non-null value.`);
// }

// NOTE: We explicitly type out what T extends instead of any so that
// util.flatten on a nested array of number doesn't try to infer T as a
// number[][], causing us to explicitly type util.flatten<number>().
/**
 *  Flattens an arbitrarily nested array.
 *
 * ```js
 * const a = [[1, 2], [3, 4], [5, [6, [7]]]];
 * const flat = tf.util.flatten(a);
 * console.log(flat);
 * ```
 *
 *  @param arr The nested array to flatten.
 *  @param result The destination array which holds the elements.
 *  @param skipTypedArray If true, avoids flattening the typed arrays. Defaults
 *      to false.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
List<T> flatten<T extends Object>(
  Object arr, {
  List<T>? result,
  bool skipTypedArray = false,
}) {
  // number|boolean|string|Promise<number>|TypedArray

  result ??= [];

  if (arr is List || arr is TypedData && !skipTypedArray) {
    arr = arr as List;
    for (int i = 0; i < arr.length; ++i) {
      flatten(arr[i], result: result, skipTypedArray: skipTypedArray);
    }
  } else {
    result.add(arr as T);
  }
  return result;
}

/**
 * Returns the size (number of elements) of the tensor given its shape.
 *
 * ```js
 * const shape = [3, 4, 2];
 * const size = tf.util.sizeFromShape(shape);
 * console.log(size);
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
int sizeFromShape(List<int> shape) {
  if (shape.length == 0) {
    // Scalar.
    return 1;
  }
  int size = shape[0];
  for (int i = 1; i < shape.length; i++) {
    size *= shape[i];
  }
  return size;
}

bool isScalarShape(List<int> shape) {
  return shape.length == 0;
}

bool arraysEqual(List? n1, List? n2) {
  if (n1 == n2) {
    return true;
  }
  if (n1 == null || n2 == null) {
    return false;
  }

  if (n1.length != n2.length) {
    return false;
  }
  for (int i = 0; i < n1.length; i++) {
    if (n1[i] != n2[i]) {
      return false;
    }
  }
  return true;
}

// export function isInt(a: number): boolean {
//   return a % 1 === 0;
// }

// export function tanh(x: number): number {
//   // tslint:disable-next-line:no-any
//   if ((Math as any).tanh != null) {
//     // tslint:disable-next-line:no-any
//     return (Math as any).tanh(x);
//   }
//   if (x === Infinity) {
//     return 1;
//   } else if (x === -Infinity) {
//     return -1;
//   } else {
//     const e2x = Math.exp(2 * x);
//     return (e2x - 1) / (e2x + 1);
//   }
// }

// export function sizeToSquarishShape(size: number): [number, number] {
//   const width = Math.ceil(Math.sqrt(size));
//   return [width, Math.ceil(size / width)];
// }

// /**
//  * Creates a new array with randomized indicies to a given quantity.
//  *
//  * ```js
//  * const randomTen = tf.util.createShuffledIndices(10);
//  * console.log(randomTen);
//  * ```
//  *
//  * @param number Quantity of how many shuffled indicies to create.
//  *
//  * @doc {heading: 'Util', namespace: 'util'}
//  */
// export function createShuffledIndices(n: number): Uint32Array {
//   const shuffledIndices = new Uint32Array(n);
//   for (let i = 0; i < n; ++i) {
//     shuffledIndices[i] = i;
//   }
//   shuffle(shuffledIndices);
//   return shuffledIndices;
// }

String rightPad(String a, int size) {
  if (size <= a.length) {
    return a;
  }
  return a + ' ' * (size - a.length);
}

// export function repeatedTry(
//     checkFn: () => boolean, delayFn = (counter: number) => 0,
//     maxCounter?: number): Promise<void> {
//   return new Promise<void>((resolve, reject) => {
//     let tryCount = 0;

//     const tryFn = () => {
//       if (checkFn()) {
//         resolve();
//         return;
//       }

//       tryCount++;

//       const nextBackoff = delayFn(tryCount);

//       if (maxCounter != null && tryCount >= maxCounter) {
//         reject();
//         return;
//       }
//       setTimeout(tryFn, nextBackoff);
//     };

//     tryFn();
//   });
// }

/**
 * Given the full size of the array and a shape that may contain -1 as the
 * implicit dimension, returns the inferred shape where -1 is replaced.
 * E.g. For shape=[2, -1, 3] and size=24, it will return [2, 4, 3].
 *
 * @param shape The shape, which may contain -1 in some dimension.
 * @param size The full size (number of elements) of the array.
 * @return The inferred shape where -1 is replaced with the inferred size.
 */
List<int> inferFromImplicitShape(List<int> shape, int size) {
  int shapeProd = 1;
  int implicitIdx = -1;

  for (int i = 0; i < shape.length; ++i) {
    if (shape[i] >= 0) {
      shapeProd *= shape[i];
    } else if (shape[i] == -1) {
      if (implicitIdx != -1) {
        throw Exception('Shapes can only have 1 implicit size. ' +
            'Found -1 at dim ${implicitIdx} and dim ${i}');
      }
      implicitIdx = i;
    } else if (shape[i] < 0) {
      throw Exception('Shapes can not be < 0. Found ${shape[i]} at dim ${i}');
    }
  }

  if (implicitIdx == -1) {
    if (size > 0 && size != shapeProd) {
      throw Exception('Size(${size}) must match the product of shape ${shape}');
    }
    return shape;
  }

  if (shapeProd == 0) {
    throw Exception('Cannot infer the missing size in [${shape}] when ' +
        'there are 0 elements');
  }
  if (size % shapeProd != 0) {
    throw Exception("The implicit shape can't be a fractional number. " +
        'Got ${size} / ${shapeProd}');
  }

  final newShape = [...shape];
  newShape[implicitIdx] = size ~/ shapeProd;
  return newShape;
}

List<int> parseAxisParam(List<int> axis, List<int> shape) {
  final rank = shape.length;

  // Normalize input
  // axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);

  // Check for valid range
  assert(
      axis.every((ax) => ax >= -rank && ax < rank),
      () =>
          'All values in axis param must be in range [-${rank}, ${rank}) but ' +
          'got axis ${axis}');

  // Check for only integers
  assert(
      axis.every((ax) => ax is int),
      () =>
          'All values in axis param must be integers but ' +
          'got axis ${axis}');

  // Handle negative axis.
  return axis.map((a) => a < 0 ? rank + a : a).toList();
}

class SqueezedShape {
  final Shape newShape;
  final List<int> keptDims;

  SqueezedShape({
    required this.newShape,
    required this.keptDims,
  });
}

/** Reduces the shape by removing all dimensions of shape 1. */
SqueezedShape squeezeShape(Shape shape, List<int>? axis) {
  final List<int> newShape = [];
  final List<int> keptDims = [];
  final isEmptyArray = axis != null && axis is List && axis.length == 0;
  final axes = (axis == null || isEmptyArray)
      ? null
      : (parseAxisParam(axis, shape)..sort());
  int j = 0;
  for (int i = 0; i < shape.length; ++i) {
    if (axes != null) {
      if (axes[j] == i && shape[i] != 1) {
        throw Exception(
            "Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1");
      }
      if ((axes[j] == null || axes[j] > i) && shape[i] == 1) {
        newShape.add(shape[i]);
        keptDims.add(i);
      }
      if (axes[j] <= i) {
        j++;
      }
    }
    if (shape[i] != 1) {
      newShape.add(shape[i]);
      keptDims.add(i);
    }
  }
  return SqueezedShape(newShape: newShape, keptDims: keptDims);
}

List getTypedArrayFromDType<D extends DataType>(D dtype, int size) {
  //DataTypeMap[D]
  final List values;
  if (dtype == null || dtype == 'float32') {
    values = Float32List(size);
  } else if (dtype == 'int32') {
    values = Int32List(size);
  } else if (dtype == 'bool') {
    values = Uint8List(size);
  } else {
    throw Exception('Unknown data type ${dtype}');
  }
  return values;
}

List getArrayFromDType<D extends DataType>(D dtype, int size) {
  // DataTypeMap[D]
  final List values;
  if (dtype == null || dtype == 'float32') {
    values = Float32List(size);
  } else if (dtype == 'int32') {
    values = Int32List(size);
  } else if (dtype == 'bool') {
    values = Uint8List(size);
  } else if (dtype == 'string') {
    values = List.filled(size, '');
  } else {
    throw Exception('Unknown data type ${dtype}');
  }
  return values;
}

void checkConversionForErrors<D extends DataType>(
  List<num> vals,
  D dtype,
) {
  for (int i = 0; i < vals.length; i++) {
    final num_ = vals[i];
    if (num_.isNaN || num_.isInfinite) {
      throw Exception(
          'A tensor of type ${dtype} being uploaded contains ${num_}.');
    }
  }
}

/** Returns true if the dtype is valid. */
bool isValidDtype(DataType dtype) {
  return dtype == 'bool' ||
      dtype == 'complex64' ||
      dtype == 'float32' ||
      dtype == 'int32' ||
      dtype == 'string';
}

/**
 * Returns true if the new type can't encode the old type without loss of
 * precision.
 */
bool hasEncodingLoss(DataType oldType, DataType newType) {
  if (newType == 'complex64') {
    return false;
  }
  if (newType == 'float32' && oldType != 'complex64') {
    return false;
  }
  if (newType == 'int32' && oldType != 'float32' && oldType != 'complex64') {
    return false;
  }
  if (newType == 'bool' && oldType == 'bool') {
    return false;
  }
  return true;
}

/**
 * Returns the current high-resolution time in milliseconds relative to an
 * arbitrary time in the past. It works across different platforms (node.js,
 * browsers).
 *
 * ```js
 * console.log(tf.util.now());
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
int now() {
  return DateTime.now().millisecondsSinceEpoch;
}

bool isTypedArray(Object? a) {
  return a is Float32List ||
      a is Int32List ||
      a is Uint8List ||
      a is Uint8ClampedList;
}

int bytesPerElement(DataType dtype) {
  if (dtype == 'float32' || dtype == 'int32') {
    return 4;
  } else if (dtype == 'complex64') {
    return 8;
  } else if (dtype == 'bool') {
    return 1;
  } else {
    throw Exception('Unknown dtype ${dtype}');
  }
}

/**
 * Returns the approximate number of bytes allocated in the string array - 2
 * bytes per character. Computing the exact bytes for a native string in JS is
 * not possible since it depends on the encoding of the html page that serves
 * the website.
 */
int bytesFromStringArray(List<Uint8List>? arr) {
  if (arr == null) {
    return 0;
  }
  int bytes = 0;
  arr.forEach((x) => bytes += x.length);
  return bytes;
}

// /** Returns true if the value is a string. */
// export function isString(value: {}): value is string {
//   return typeof value === 'string' || value instanceof String;
// }

// export function isBoolean(value: {}): boolean {
//   return typeof value === 'boolean';
// }

// export function isNumber(value: {}): boolean {
//   return typeof value === 'number';
// }

DataType inferDtype(Object values) {
  if (values is List) {
    return inferDtype(values[0]);
  }
  if (values is Float32List) {
    return 'float32';
  } else if (values is Int32List ||
      values is Uint8List ||
      values is Uint8ClampedList) {
    return 'int32';
  } else if (values is num) {
    return 'float32';
  } else if (values is String) {
    return 'string';
  } else if (values is bool) {
    return 'bool';
  }
  return 'float32';
}

// export function isFunction(f: Function) {
//   return !!(f && f.constructor && f.call && f.apply);
// }

int nearestDivisor(int size, int start) {
  for (int i = start; i < size; ++i) {
    if (size % i == 0) {
      return i;
    }
  }
  return size;
}

List<int> computeStrides(List<int> shape) {
  final rank = shape.length;
  if (rank < 2) {
    return [];
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  final strides = List.filled(rank - 1, 0);
  strides[rank - 2] = shape[rank - 1];
  for (int i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

List createNestedArray(
  int offset,
  List<int> shape,
  List a, {
  bool isComplex = false,
}) {
  final ret = [];
  if (shape.length == 1) {
    final d = shape[0] * (isComplex ? 2 : 1);
    for (int i = 0; i < d; i++) {
      ret[i] = a[offset + i];
    }
  } else {
    final d = shape[0];
    final rest = shape.sublist(1);
    final len = rest.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
    for (int i = 0; i < d; i++) {
      ret[i] = createNestedArray(
        offset + i * len,
        rest,
        a,
        isComplex: isComplex,
      );
    }
  }
  return ret;
}

// Provide a nested array of TypedArray in given shape.
Object toNestedArray(List<int> shape, List a, {bool isComplex = false}) {
  if (shape.length == 0) {
    // Scalar type should return a single number.
    return a[0];
  }
  final size = shape.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
  if (size == 0) {
    // A tensor with shape zero should be turned into empty list.
    return [];
  }
  if (size != a.length) {
    throw Exception(
        "[${shape}] does not match the input size ${a.length}${isComplex ? ' for a complex tensor' : ''}.");
  }

  return createNestedArray(0, shape, a, isComplex: isComplex);
}

List<num> makeOnesTypedArray<D extends DataType>(int size, D dtype) {
  final array = makeZerosTypedArray(size, dtype);
  for (int i = 0; i < array.length; i++) {
    array[i] = 1;
  }
  return array;
}

List<num> makeZerosTypedArray<D extends DataType>(int size, D dtype) {
  if (dtype == null || dtype == 'float32' || dtype == 'complex64') {
    return Float32List(size);
  } else if (dtype == 'int32') {
    return Int32List(size);
  } else if (dtype == 'bool') {
    return Uint8List(size);
  } else {
    throw Exception('Unknown data type ${dtype}');
  }
}

// /**
//  * Make nested `TypedArray` filled with zeros.
//  * @param shape The shape information for the nested array.
//  * @param dtype dtype of the array element.
//  */
// export function makeZerosNestedTypedArray<D extends DataType>(
//     shape: number[], dtype: D) {
//   const size = shape.reduce((prev, curr) => prev * curr, 1);
//   if (dtype == null || dtype === 'float32') {
//     return toNestedArray(shape, new Float32Array(size));
//   } else if (dtype === 'int32') {
//     return toNestedArray(shape, new Int32Array(size));
//   } else if (dtype === 'bool') {
//     return toNestedArray(shape, new Uint8Array(size));
//   } else {
//     throw new Error(`Unknown data type ${dtype}`);
//   }
// }

void assertNonNegativeIntegerDimensions(List<int> shape) {
  shape.forEach((dimSize) {
    assert_(
      dimSize == dimSize.toInt() && dimSize >= 0,
      () =>
          'Tensor must have a shape comprised of positive integers but got ' +
          'shape [${shape}].',
    );
  });
}

// /**
//  * Computes flat index for a given location (multidimentionsal index) in a
//  * Tensor/multidimensional array.
//  *
//  * @param locs Location in the tensor.
//  * @param rank Rank of the tensor.
//  * @param strides Tensor strides.
//  */
// export function locToIndex(
//     locs: number[], rank: number, strides: number[]): number {
//   if (rank === 0) {
//     return 0;
//   } else if (rank === 1) {
//     return locs[0];
//   }
//   let index = locs[locs.length - 1];
//   for (let i = 0; i < locs.length - 1; ++i) {
//     index += strides[i] * locs[i];
//   }
//   return index;
// }

// /**
//  * Computes the location (multidimensional index) in a tensor/multidimentional
//  * array for a given flat index.
//  *
//  * @param index Index in flat array.
//  * @param rank Rank of tensor.
//  * @param strides Strides of tensor.
//  */
// export function indexToLoc(
//     index: number, rank: number, strides: number[]): number[] {
//   if (rank === 0) {
//     return [];
//   } else if (rank === 1) {
//     return [index];
//   }
//   const locs: number[] = new Array(rank);
//   for (let i = 0; i < locs.length - 1; ++i) {
//     locs[i] = Math.floor(index / strides[i]);
//     index -= locs[i] * strides[i];
//   }
//   locs[locs.length - 1] = index;
//   return locs;
// }

// /**
//  * This method asserts whether an object is a Promise instance.
//  * @param object
//  */
// // tslint:disable-next-line: no-any
// export function isPromise(object: any): object is Promise<unknown> {
//   //  We chose to not use 'obj instanceOf Promise' for two reasons:
//   //  1. It only reliably works for es6 Promise, not other Promise
//   //  implementations.
//   //  2. It doesn't work with framework that uses zone.js. zone.js monkey patch
//   //  the async calls, so it is possible the obj (patched) is comparing to a
//   //  pre-patched Promise.
//   return object && object.then && typeof object.then === 'function';
// }

/**
 * Create typed array for scalar value. Used for storing in `DataStorage`.
 */
BackendValues createScalarValue(DataType value, DataType dtype) {
  if (dtype == 'string') {
    return encodeString(value);
  }

  return toTypedArray([value], dtype) as BackendValues;
}

bool noConversionNeeded(TensorLike a, DataType dtype) {
  return (a is Float32List && dtype == 'float32') ||
      (a is Int32List && dtype == 'int32') ||
      (a is Uint8List && dtype == 'bool');
}

TypedData toTypedArray(TensorLike a, DataType dtype) {
  if (dtype == 'string') {
    throw Exception('Cannot convert a string[] to a TypedArray');
  }
  if (a is List) {
    a = flatten(a);
  }

  if (env().getBool('DEBUG')) {
    checkConversionForErrors((a as List).cast(), dtype);
  }
  if (noConversionNeeded(a, dtype)) {
    return a as TypedData;
  }
  a = a as List;
  if (dtype == null || dtype == 'float32' || dtype == 'complex64') {
    return Float32List.fromList(
      a is List<double> ? a : a.map((e) => (e as num).toDouble()).toList(),
    );
  } else if (dtype == 'int32') {
    return Int32List.fromList(a.cast());
  } else if (dtype == 'bool') {
    final bool_ = Uint8List((a).length);
    for (int i = 0; i < bool_.length; ++i) {
      if (a[i] != 0) {
        bool_[i] = 1;
      }
    }
    return bool_;
  } else {
    throw Exception('Unknown data type ${dtype}');
  }
}

/**
 * Encodes the provided string into bytes using the provided encoding scheme.
 *
 * @param s The string to encode.
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
Uint8List encodeString(String s, {String encoding = 'utf-8'}) {
  return utf8.encoder.convert(s);
}

/**
 * Decodes the provided bytes into a string using the provided encoding scheme.
 * @param bytes The bytes to decode.
 *
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
String decodeString(Uint8List bytes, {String encoding = 'utf-8'}) {
  return utf8.decode(bytes);
}
