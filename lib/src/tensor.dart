// /**
//  * @license
//  * Copyright 2017 Google LLC. All Rights Reserved.
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

// ignore_for_file: unnecessary_this, constant_identifier_names

import 'dart:typed_data';
import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/src/backend.dart';

import 'util_base.dart' as util;

class TensorInfo {
  final DataId dataId;
  final List<int> shape;
  final DataType dtype;

  TensorInfo({
    required this.dataId,
    required this.shape,
    required this.dtype,
  });
}

typedef DataId = Map; // object instead of {} to force non-primitive.
typedef DataType = String;
typedef BackendValues
    = List; //  Float32Array|Int32Array|Uint8Array|Uint8Array[];
typedef DataValues
    = List; // Float32Array|Int32Array|Uint8Array|Float32Array|string[];

typedef NamedVariableMap = Map<String, Variable>;

typedef GradSaveFunc = void Function(List<Tensor> save);

typedef NamedTensorMap = Map<String, Tensor>;

/**
 * @docalias void|number|string|TypedArray|Tensor|Tensor[]|{[key:
 * string]:Tensor|number|string}
 */
typedef TensorContainer = Object?;
// void|Tensor|string|number|boolean|TensorContainerObject|
// TensorContainerArray|Float32Array|Int32Array|Uint8Array;

typedef TensorContainerObject = Map<String, TensorContainer>;

typedef TensorContainerArray = List<TensorContainer>;

class TensorList extends DelegatingList<Tensor> with Tensors {
  TensorList(List<Tensor<Rank>> base) : super(base);
}

class Tensors {
  O match<O>(
    O Function(Tensor tensor) tensor,
    O Function(TensorList list) list,
  ) =>
      this is TensorList ? list(this as TensorList) : tensor(this as Tensor);
}

class ListOrVal<T> {
  final Object? _value;
  final bool isList;

  ListOrVal.list(List<T> this._value) : isList = true;
  ListOrVal.val(T this._value) : isList = false;

  List<T> get asList => isList ? _value as List<T> : [_value as T];
  T? get asVal => !isList ? _value as T : null;

  O match<O>(
    O Function(T val) val,
    O Function(List<T> list) list,
  ) =>
      isList ? list(_value as List<T>) : val(_value as T);
}

// import {getGlobal} from './global_util';
// import {tensorToString} from './tensor_format';
// import {ArrayMap, BackendValues, DataType, DataTypeMap, DataValues, NumericDataType, Rank, ShapeMap, SingleValueMap, TypedArray} from './types';
// import * as util from './util';
// import {computeStrides, toNestedArray} from './util';

enum Rank { R0, R1, R2, R3, R4, R5, R6 }

class TensorData<D extends DataType> {
  DataId? dataId;
  List? values; // DataTypeMap[D]
}

// This interface mimics KernelBackend (in backend.ts), which would create a
// circular dependency if imported.
class Backend {}

/**
 * A mutable object, similar to `tf.Tensor`, that allows users to set values
 * at locations before converting to an immutable `tf.Tensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
class TensorBuffer<R extends Rank, D extends DataType> {
  final int size;
  final List<int> shape; // ShapeMap[D]
  late final List<int> strides;
  late final List values; // DataTypeMap[D]
  final D dtype;

  TensorBuffer(this.shape, this.dtype, [List? values])
      : size = util.sizeFromShape(shape),
        strides = util.computeStrides(shape) {
    if (values != null) {
      final n = values.length;
      util.assert_(
          n == this.size,
          () =>
              "Length of values '${n}' does not match the size " +
              "inferred by the shape '${this.size}'.");
    }
    if (dtype == 'complex64') {
      throw Exception(
          'complex64 dtype TensorBuffers are not supported. Please create ' +
              'a TensorBuffer for the real and imaginary parts separately and ' +
              'call tf.complex(real, imag).');
    }
    this.values = values ?? util.getArrayFromDType(dtype, this.size);
  }

  /**
   * Sets a value in the buffer at a given location.
   *
   * @param value The value to set.
   * @param locs  The location indices.
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
  void set(Object value, List<int> locs) {
    if (locs.length == 0) {
      locs = [0];
    }
    util.assert_(
        locs.length == this.rank,
        () =>
            'The number of provided coordinates (${locs.length}) must ' +
            'match the rank (${this.rank})');

    final index = this.locToIndex(locs);
    this.values[index] = value;
  }

  /**
   * Returns the value in the buffer at the provided location.
   *
   * @param locs The location indices.
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
  Object get(List<int> locs) {
    // SingleValueMap[D]
    if (locs.length == 0) {
      locs = [0];
    }
    int i = 0;
    for (final loc in locs) {
      if (loc < 0 || loc >= this.shape[i]) {
        final msg = 'Requested out of range element at ${locs}. ' +
            '  Buffer shape=${this.shape}';
        throw Exception(msg);
      }
      i++;
    }
    int index = locs[locs.length - 1];
    for (int i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.values[index];
  }

  int locToIndex(List<int> locs) {
    if (this.rank == 0) {
      return 0;
    } else if (this.rank == 1) {
      return locs[0];
    }
    int index = locs[locs.length - 1];
    for (int i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  List<int> indexToLoc(int index) {
    if (this.rank == 0) {
      return [];
    } else if (this.rank == 1) {
      return [index];
    }
    final locs = <int>[];
    for (int i = 0; i < this.shape.length - 1; ++i) {
      locs.add((index / this.strides[i]).floor());
      index -= locs[i] * this.strides[i];
    }
    locs.add(index);
    return locs;
  }

  int get rank {
    return this.shape.length;
  }

  /**
   * Creates an immutable `tf.Tensor` object from the buffer.
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
  Tensor<R> toTensor() {
    return trackerFn().makeTensor(this.values, this.shape, this.dtype)
        as Tensor<R>;
  }
}

class DataToGPUWebGLOption {
  final List<int>? customTexShape;

  DataToGPUWebGLOption(this.customTexShape);
}

typedef DataToGPUOptions = DataToGPUWebGLOption;
typedef WebGLTexture = Object;

class GPUData {
  Tensor tensorRef;
  WebGLTexture texture;
  List<int>? texShape;

  GPUData({
    required this.tensorRef,
    required this.texture,
    this.texShape,
  });
}

abstract class TensorTracker {
  Tensor makeTensor(
    DataValues values,
    List<int> shape,
    DataType dtype, [
    KernelBackend? backend,
  ]);
  Variable makeVariable(
    Tensor initialValue, {
    bool trainable,
    String? name,
    DataType? dtype,
  });
  void incRef(Tensor a, KernelBackend? backend);
  void disposeTensor(Tensor t);
  void disposeVariable(Variable v);
  Future<BackendValues> read(DataId dataId);
  BackendValues readSync(DataId dataId);
}

/**
 * The Tensor class calls into this handler to delegate chaining operations.
 */
class OpHandler {
  final T Function<T extends Tensor>(T x, DataType dtype) cast;
  final TensorBuffer<R, D> Function<R extends Rank, D extends DataType>(
    List<int> shape,
    D dtype,
    List? values,
  ) buffer;
  final void Function<T extends Tensor>(T x, {bool verbose}) print;
  final T Function<T extends Tensor>(T x) clone;

  OpHandler(this.buffer, this.cast, this.clone, this.print);
  // TODO(yassogba) bring reshape back?
}

// For tracking tensor creation and disposal.
late TensorTracker Function() trackerFn;
// Used by chaining methods to call into ops.
late OpHandler opHandler;
// Used to warn about deprecated methods.
void Function(String)? deprecationWarningFn = null;
// This here so that we can use this method on dev branches and keep the
// functionality at master.
// tslint:disable-next-line:no-unused-expression
// [deprecationWarningFn];

/**
 * An external consumer can register itself as the tensor tracker. This way
 * the Tensor class can notify the tracker for every tensor created and
 * disposed.
 */
void setTensorTracker(TensorTracker Function() fn) {
  trackerFn = fn;
}

/**
 * An external consumer can register itself as the op handler. This way the
 * Tensor class can have chaining methods that call into ops via the op
 * handler.
 */
void setOpHandler(OpHandler handler) {
  opHandler = handler;
}

/**
 * Sets the deprecation warning function to be used by this file. This way the
 * Tensor class can be a leaf but still use the environment.
 */
void setDeprecationWarningFn(void Function(String) fn) {
  deprecationWarningFn = fn;
}

// /**
//  * We wrap data id since we use weak map to avoid memory leaks.
//  * Since we have our own memory management, we have a reference counter
//  * mapping a tensor to its data, so there is always a pointer (even if that
//  * data is otherwise garbage collectable).
//  * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
//  * Global_Objects/WeakMap
//  */
// export type DataId = object;  // object instead of {} to force non-primitive.

// Declare this namespace to make Tensor class augmentation work in google3.

/**
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * For performance reasons, functions that create tensors do not necessarily
 * perform a copy of the data passed to them (e.g. if the data is passed as a
 * `Float32Array`), and changes to the data will change the tensor. This is not
 * a feature and is not supported. To avoid this behavior, use the tensor before
 * changing the input data or create a copy with `copy = tf.add(yourTensor, 0)`.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
class Tensor<R extends Rank> with Tensors implements TensorInfo {
  /** Unique id of this tensor. */
  final int id;
  /**
   * Id of the bucket holding the data for this tensor. Multiple arrays can
   * point to the same bucket (e.g. when calling array.reshape()).
   */
  DataId dataId;
  /** The shape of the tensor. */
  late final List<int> shape; // ShapeMap[R]
  /** Number of elements in the tensor. */
  late final int size;
  /** The data type for the array. */
  late final DataType dtype;
  /** The rank type for the array (see `Rank` enum). */
  late final String rankType; // TODO: R

  /** Whether this tensor has been globally kept. */
  bool kept = false;
  /** The id of the scope this tensor is being tracked in. */
  int? scopeId;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated/\
   * numpy.ndarray.strides.html
   */
  late final List<int> strides;

  Tensor(this.shape, DataType? dtype, this.dataId, this.id) {
    // TODO: this.shape = shape.slice() as ShapeMap[R];
    this.dtype = dtype ?? 'float32';
    this.size = util.sizeFromShape(shape);
    this.strides = util.computeStrides(shape);
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
  }

  int get rank {
    return this.shape.length;
  }

  /**
   * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Future<TensorBuffer<R, D>> buffer<D extends DataType>() async {
    final vals = await this.data<D>();
    return opHandler.buffer(this.shape, this.dtype as D, vals);
  }

  /**
   * Returns a `tf.TensorBuffer` that holds the underlying data.
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  TensorBuffer<R, D> bufferSync<D extends DataType>() {
    return opHandler.buffer(this.shape, this.dtype as D, this.dataSync());
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * asynchronously.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Future<Object> array() async {
    // TODO: should be a list?
    // TODO: ArrayMap[R]
    final vals = await this.data();
    return util.toNestedArray(
      this.shape,
      vals,
      isComplex: this.dtype == 'complex64',
    );
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * synchronously.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Object arraySync() {
    // TODO: should be a list?
    // TODO: ArrayMap[R]
    return util.toNestedArray(
      this.shape,
      this.dataSync(),
      isComplex: this.dtype == 'complex64',
    );
  }

  /**
   * Asynchronously downloads the values from the `tf.Tensor`. Returns a
   * promise of `TypedArray` that resolves when the computation has finished.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Future<List> data<D extends DataType>() async {
    //  TODO: DataTypeMap[D]
    this.throwIfDisposed();
    final data = trackerFn().read(this.dataId);
    if (this.dtype == 'string') {
      final bytes = await data as List<Uint8List>;
      try {
        return bytes.map((b) => util.decodeString(b)).toList();
      } catch (_) {
        throw Exception('Failed to decode the string bytes into utf-8. ' +
            'To get the original bytes, call tensor.bytes().');
      }
    }
    return data;
  }

  /**
   * Synchronously downloads the values from the `tf.Tensor`. This blocks the
   * UI thread until the values are ready, which can cause performance issues.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  List dataSync<D extends DataType>() {
    //  TODO: DataTypeMap[D]
    this.throwIfDisposed();
    final data = trackerFn().readSync(this.dataId);
    if (this.dtype == 'string') {
      try {
        return (data as List<Uint8List>)
            .map((b) => util.decodeString(b))
            .toList();
      } catch (_) {
        throw Exception('Failed to decode the string bytes into utf-8. ' +
            'To get the original bytes, call tensor.bytes().');
      }
    }
    return data;
  }

  /** Returns the underlying bytes of the tensor's data. */
  Future<ListOrVal<Uint8List>> bytes() async {
    this.throwIfDisposed();
    final data = await trackerFn().read(this.dataId);
    if (this.dtype == 'string') {
      return ListOrVal.list(data as List<Uint8List>);
    } else {
      return ListOrVal.val(Uint8List.view((data as TypedData).buffer));
    }
  }

  /**
   * Disposes `tf.Tensor` from memory.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  void dispose() {
    if (this.isDisposed) {
      return;
    }
    trackerFn().disposeTensor(this);
    this._isDisposedInternal = true;
  }

  bool _isDisposedInternal = false;
  bool get isDisposed {
    return this._isDisposedInternal;
  }

  throwIfDisposed() {
    if (this.isDisposed) {
      throw Exception('Tensor is disposed.');
    }
  }

  /**
   * Prints the `tf.Tensor`. See `tf.print` for details.
   *
   * @param verbose Whether to print verbose information about the tensor,
   *    including dtype and size.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  void print({bool verbose = false}) {
    return opHandler.print(this, verbose: verbose);
  }

  /**
   * Returns a copy of the tensor. See `tf.clone` for details.
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  T clone<T extends Tensor>(T d) {
    d.throwIfDisposed();
    return opHandler.clone(d);
  }

  /**
   * Returns a human-readable description of the tensor. Useful for logging.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  String toString({bool verbose = false}) {
    final vals = this.dataSync();
    return '{vals: $vals, shape: $shape, dtype: $dtype}';
    // TODO:
    // return tensorToString(vals, this.shape, this.dtype, verbose);
  }

  T cast<T extends Tensor>(DataType dtype) {
    this.throwIfDisposed();
    return opHandler.cast(this as T, dtype);
  }

  Variable<R> variable({bool trainable = true, String? name, DataType? dtype}) {
    this.throwIfDisposed();
    return trackerFn().makeVariable(
      this,
      trainable: trainable,
      name: name,
      dtype: dtype,
    ) as Variable<R>;
  }
}

// Object.defineProperty(Tensor, Symbol.hasInstance, {
//   value: (instance: Tensor) => {
//     // Implementation note: we should use properties of the object that will be
//     // defined before the constructor body has finished executing (methods).
//     // This is because when this code is transpiled by babel, babel will call
//     // classCallCheck before the constructor body is run.
//     // See https://github.com/tensorflow/tfjs/issues/3384 for backstory.
//     return !!instance && instance.data != null && instance.dataSync != null &&
//         instance.throwIfDisposed != null;
//   }
// });

// export function getGlobalTensorClass() {
//   // Use getGlobal so that we can augment the Tensor class across package
//   // boundaries becase the node resolution alg may result in different modules
//   // being returned for this file depending on the path they are loaded from.
//   return getGlobal('Tensor', () => {
//     return Tensor;
//   });
// }

// // Global side effect. Cache global reference to Tensor class
// getGlobalTensorClass();

// export interface NumericTensor<R extends Rank = Rank> extends Tensor<R> {
//   dtype: NumericDataType;
//   dataSync<D extends DataType = NumericDataType>(): DataTypeMap[D];
//   data<D extends DataType = NumericDataType>(): Promise<DataTypeMap[D]>;
// }

// export interface StringTensor<R extends Rank = Rank> extends Tensor<R> {
//   dtype: 'string';
//   dataSync<D extends DataType = 'string'>(): DataTypeMap[D];
//   data<D extends DataType = 'string'>(): Promise<DataTypeMap[D]>;
// }

// /** @doclink Tensor */
// typedef Scalar = Tensor<Rank.R0>;
// /** @doclink Tensor */
// export type Tensor1D = Tensor<Rank.R1>;
// /** @doclink Tensor */
// export type Tensor2D = Tensor<Rank.R2>;
// /** @doclink Tensor */
// export type Tensor3D = Tensor<Rank.R3>;
// /** @doclink Tensor */
// export type Tensor4D = Tensor<Rank.R4>;
// /** @doclink Tensor */
// export type Tensor5D = Tensor<Rank.R5>;
// /** @doclink Tensor */
// export type Tensor6D = Tensor<Rank.R6>;

/**
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
class Variable<R extends Rank> extends Tensor<R> {
  final String name;
  final bool trainable;

  Variable(Tensor<R> initialValue, this.trainable, this.name, int tensorId)
      : super(initialValue.shape, initialValue.dtype, initialValue.dataId,
            tensorId);

  /**
   * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
   * the same shape and dtype as the old `tf.Tensor`.
   *
   * @param newValue New tensor to be assigned to this variable.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  void assign(Tensor<R> newValue) {
    if (newValue.dtype != this.dtype) {
      throw Exception('dtype of the new value (${newValue.dtype}) and ' +
          'previous value (${this.dtype}) must match');
    }
    if (!util.arraysEqual(newValue.shape, this.shape)) {
      throw Exception('shape of the new value (${newValue.shape}) and ' +
          'previous value (${this.shape}) must match');
    }
    trackerFn().disposeTensor(this);
    this.dataId = newValue.dataId;
    trackerFn().incRef(this, null /* backend */);
  }

  void dispose() {
    trackerFn().disposeVariable(this);
    this._isDisposedInternal = true;
  }
}

// Object.defineProperty(Variable, Symbol.hasInstance, {
//   value: (instance: Variable) => {
//     return instance instanceof Tensor && instance.assign != null &&
//         instance.assign instanceof Function;
//   }
// });