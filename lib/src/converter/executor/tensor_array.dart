import 'package:tensorflow_wasm/src/converter/executor/tensor_utils.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'dart:math' as math;

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

// import {concat, DataType, keep, reshape, scalar, slice, stack, Tensor, tensor, tidy, unstack} from '@tensorflow/tfjs-core';

// import {assertShapesMatchAllowUndefinedSize} from './tensor_utils';

class TensorWithState {
  final Tensor tensor;
  bool? written;
  bool? read;
  bool? cleared;

  TensorWithState({
    required this.tensor,
    this.written,
    this.read,
    this.cleared,
  });
}

/**
 * The TensorArray object keeps an array of Tensors.  It
 * allows reading from the array and writing to the array.
 */
class TensorArray {
  List<TensorWithState> tensors = []; // private
  bool closed_ = false; // private
  final Tensor idTensor;
  final String name;
  final DataType dtype;
  int maxSize; // private
  List<int> elementShape; // private
  final bool identicalElementShapes;
  final bool dynamicSize;
  final bool clearAfterRead;

  TensorArray(
    this.name,
    this.dtype,
    this.maxSize,
    this.elementShape,
    this.identicalElementShapes,
    this.dynamicSize,
    this.clearAfterRead,
  ) : idTensor = scalar(0) {
    keep(this.idTensor);
  }

  get id {
    return this.idTensor.id;
  }

  get closed {
    return this.closed_;
  }

  /**
   * Dispose the tensors and idTensor and mark the TensoryArray as closed.
   */
  clearAndClose(Set<int>? keepIds) {
    this.tensors.forEach((tensor) {
      if (keepIds == null || !keepIds.contains(tensor.tensor.id)) {
        tensor.tensor.dispose();
      }
    });
    this.tensors = [];
    this.closed_ = true;
    this.idTensor.dispose();
  }

  int size() {
    return this.tensors.length;
  }

  /**
   * Read the value at location index in the TensorArray.
   * @param index Number the index to read from.
   */
  Tensor read(int index) {
    if (this.closed_) {
      throw Exception('TensorArray ${this.name} has already been closed.');
    }

    if (index < 0 || index >= this.size()) {
      throw Exception(
          'Tried to read from index ${index}, but array size is: ${this.size()}');
    }

    final tensorWithState = this.tensors[index];
    if (tensorWithState.cleared == true) {
      throw Exception(
          'TensorArray ${this.name}: Could not read index ${index} twice because it was cleared after a previous read ' +
              '(perhaps try setting clear_after_read = false?).');
    }

    if (this.clearAfterRead) {
      tensorWithState.cleared = true;
    }

    tensorWithState.read = true;
    return tensorWithState.tensor;
  }

  /**
   * Helper method to read multiple tensors from the specified indices.
   */
  List<Tensor> readMany(List<int> indices) {
    return indices.map((index) => this.read(index)).toList();
  }

  /**
   * Write value into the index of the TensorArray.
   * @param index number the index to write to.
   * @param tensor
   */
  write(int index, Tensor tensor) {
    if (this.closed_) {
      throw Exception('TensorArray ${this.name} has already been closed.');
    }

    if (index < 0 || !this.dynamicSize && index >= this.maxSize) {
      throw Exception(
          'Tried to write to index ${index}, but array is not resizeable and size is: ${this.maxSize}');
    }

    final t = this.tensors[index] ?? TensorWithState(tensor: tensor);

    if (tensor.dtype != this.dtype) {
      throw Exception(
          'TensorArray ${this.name}: Could not write to TensorArray index ${index}, because the value dtype is ${tensor.dtype}, but TensorArray dtype is ${this.dtype}.');
    }

    // Set the shape for the first time write to unknow shape tensor array
    if (this.size() == 0 &&
        (this.elementShape == null || this.elementShape.length == 0)) {
      this.elementShape = tensor.shape;
    }

    assertShapesMatchAllowUndefinedSize(this.elementShape, tensor.shape,
        'TensorArray ${this.name}: Could not write to TensorArray index ${index}.');

    if (t.read == true) {
      throw Exception(
          'TensorArray ${this.name}: Could not write to TensorArray index ${index}, because it has already been read.');
    }

    if (t.written == true) {
      throw Exception(
          'TensorArray ${this.name}: Could not write to TensorArray index ${index}, because it has already been written.');
    }

    keep(tensor);
    t.written = true;

    this.tensors[index] = t;
  }

  /**
   * Helper method to write multiple tensors to the specified indices.
   */
  writeMany(List<int> indices, List<Tensor> tensors) {
    if (indices.length != tensors.length) {
      throw Exception(
          'TensorArray ${this.name}: could not write multiple tensors,' +
              'because the index size: ${indices.length} is not the same as tensors size: ${tensors.length}.');
    }
    int index = 0;
    indices.forEach((i) => this.write(i, tensors[index++]));
  }

  /**
   * Return selected values in the TensorArray as a packed Tensor. All of
   * selected values must have been written and their shapes must all match.
   * @param [indices] number[] Optional. Taking values in [0, max_value). If the
   *    TensorArray is not dynamic, max_value=size(). If not specified returns
   *    all tensors in the original order.
   * @param [dtype]
   */
  Tensor gather(List<int>? indices, DataType? dtype) {
    if (dtype != null && dtype != this.dtype) {
      throw Exception(
          'TensorArray dtype is ${this.dtype} but gather requested dtype ${dtype}');
    }

    if (indices == null) {
      indices = [];
      for (int i = 0; i < this.size(); i++) {
        indices.add(i);
      }
    } else {
      indices = indices.slice(0, this.size());
    }

    if (indices!.length == 0) {
      return tensor([], [0, ...this.elementShape]);
    }

    // Read all the PersistentTensors into a vector to keep track of
    // their memory.
    final tensors = this.readMany(indices);

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensors[0].shape, 'TensorArray shape mismatch: ');

    return stack(tensors, 0);
  }

  /**
   * Return the values in the TensorArray as a concatenated Tensor.
   */
  Tensor concat(DataType? dtype) {
    if (dtype != null && dtype != this.dtype) {
      throw Exception(
          'TensorArray dtype is ${this.dtype} but concat requested dtype ${dtype}');
    }

    if (this.size() == 0) {
      return tensor([], [0, ...this.elementShape]);
    }

    final List<int> indices = List<int>.generate(this.size(), (index) => index);
    // Collect all the tensors from the tensors array.
    final tensors = this.readMany(indices);

    assertShapesMatchAllowUndefinedSize(this.elementShape, tensors[0].shape,
        'TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${tensors[0].shape})');

    return tf.concat(tensors, 0);
  }

  /**
   * Scatter the values of a Tensor in specific indices of a TensorArray.
   * @param indices nummber[] values in [0, max_value). If the
   *    TensorArray is not dynamic, max_value=size().
   * @param tensor Tensor input tensor.
   */
  scatter(List<int> indices, Tensor tensor) {
    if (tensor.dtype != this.dtype) {
      throw Exception(
          'TensorArray dtype is ${this.dtype} but tensor has dtype ${tensor.dtype}');
    }

    if (indices.length != tensor.shape[0]) {
      throw Exception(
          'Expected len(indices) == tensor.shape[0], but saw: ${indices.length} vs. ${tensor.shape[0]}');
    }

    final maxIndex = indices.reduce(math.max);

    if (!this.dynamicSize && maxIndex >= this.maxSize) {
      throw Exception(
          'Max index must be < array size (${maxIndex}  vs. ${this.maxSize})');
    }

    this.writeMany(indices, unstack(tensor, 0));
  }

  /**
   * Split the values of a Tensor into the TensorArray.
   * @param length number[] with the lengths to use when splitting value along
   *    its first dimension.
   * @param tensor Tensor, the tensor to split.
   */
  split(List<int> length, Tensor tensor) {
    if (tensor.dtype != this.dtype) {
      throw Exception(
          'TensorArray dtype is ${this.dtype} but tensor has dtype ${tensor.dtype}');
    }
    int totalLength = 0;
    final cumulativeLengths = length.map((len) {
      totalLength += len;
      return totalLength;
    }).toList();

    if (totalLength != tensor.shape[0]) {
      throw Exception('Expected sum of lengths to be equal to tensor.shape[0],'
          " but sum of lengths is ${totalLength}, and tensor's shape is: ${tensor.shape}");
    }

    if (!this.dynamicSize && length.length != this.maxSize) {
      throw Exception(
          "TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${length.length}), " +
              'and the TensorArray is not marked as dynamically resizeable');
    }

    final elementPerRow = totalLength == 0 ? 0 : tensor.size ~/ totalLength;
    final List<Tensor> tensors = [];
    tidy(() {
      tensor = reshape(tensor, [1, totalLength, elementPerRow]);
      for (int i = 0; i < length.length; ++i) {
        final previousLength = (i == 0) ? 0 : cumulativeLengths[i - 1];
        final indices = [0, previousLength, 0];
        final sizes = [1, length[i], elementPerRow];
        tensors[i] = reshape(slice(tensor, indices, sizes), this.elementShape);
      }
      return tensors;
    });
    final indices = Iterable<int>.generate(length.length).toList();
    this.writeMany(indices, tensors);
  }
}
