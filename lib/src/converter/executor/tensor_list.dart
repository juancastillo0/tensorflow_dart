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

// import {concat, DataType, keep, reshape, scalar, slice, stack, Tensor, tensor, tidy, unstack} from '@tensorflow/tfjs-core';

// import {assertShapesMatchAllowUndefinedSize, inferElementShape, mergeElementShape} from './tensor_utils';

import 'package:tensorflow_wasm/src/converter/executor/tensor_utils.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'dart:math' as math;

/**
 * TensorList stores a container of `tf.Tensor` objects, which are accessible
 * via tensors field.
 *
 * In order to get a copy of the underlying list, use the copy method:
 * ```
 *    TensorList b = a.copy();
 *    b.tensors().pushBack(t);  // This does not modify a.tensors().
 * ```
 *
 * Note that this is not a deep copy: the memory locations of the underlying
 * tensors will still point to the same locations of the corresponding tensors
 * in the original.
 */

class TensorListContainer {
  final Tensor idTensor;
  int maxNumElements;
  final List<Tensor> tensors;
  final List<int> elementShape; // int|List<int>
  final DataType elementDtype;

  get id {
    return this.idTensor.id;
  }

  /**
   *
   * @param tensors list of tensors
   * @param elementShape shape of each tensor, this can be a single number (any
   * shape is allowed) or partial shape (dim = -1).
   * @param elementDtype data type of each tensor
   * @param maxNumElements The maximum allowed size of `tensors`. Defaults to -1
   *   meaning that the size of `tensors` is unbounded.
   */
  TensorListContainer(
    this.tensors,
    this.elementShape,
    this.elementDtype, [
    int? maxNumElements,
  ])  : idTensor = scalar(0),
        maxNumElements = maxNumElements ?? -1 {
    if (tensors != null) {
      tensors.forEach((tensor) {
        if (elementDtype != tensor.dtype) {
          throw Exception(
              "Invalid data types; op elements ${elementDtype}, but list elements ${tensor.dtype}");
        }
        assertShapesMatchAllowUndefinedSize(
            elementShape, tensor.shape, 'TensorList shape mismatch: ');

        keep(tensor);
      });
    }
    keep(this.idTensor);
  }

  /**
   * Get a new TensorList containing a copy of the underlying tensor container.
   */
  TensorListContainer copy() {
    return TensorListContainer(
      [...this.tensors],
      this.elementShape,
      this.elementDtype,
    );
  }

  /**
   * Dispose the tensors and idTensor and clear the tensor list.
   */
  clearAndClose(Set<int>? keepIds) {
    this.tensors.forEach((tensor) {
      if (keepIds == null || !keepIds.contains(tensor.id)) {
        tensor.dispose();
      }
    });
    this.tensors.length = 0;
    this.idTensor.dispose();
  }

  /**
   * The size of the tensors in the tensor list.
   */
  size() {
    return this.tensors.length;
  }

  /**
   * Return a tensor that stacks a list of rank-R tf.Tensors into one rank-(R+1)
   * tf.Tensor.
   * @param elementShape shape of each tensor
   * @param elementDtype data type of each tensor
   * @param numElements the number of elements to stack
   */
  Tensor stack(List<int> elementShape, DataType elementDtype,
      [int numElements = -1]) {
    if (elementDtype != this.elementDtype) {
      throw Exception(
          "Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}");
    }
    if (numElements != -1 && this.tensors.length != numElements) {
      throw Exception(
          "Operation expected a list with ${numElements} elements but got a list with ${this.tensors.length} elements.");
    }
    assertShapesMatchAllowUndefinedSize(
        elementShape, this.elementShape, 'TensorList shape mismatch: ');
    final outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    return tidy(() {
      final reshapedTensors = this
          .tensors
          .map((tensor) => reshape(tensor, outputElementShape))
          .toList();
      return tf.stack(reshapedTensors, 0);
    });
  }

  /**
   * Pop a tensor from the end of the list.
   * @param elementShape shape of the tensor
   * @param elementDtype data type of the tensor
   */
  Tensor popBack(List<int> elementShape, DataType elementDtype) {
    if (elementDtype != this.elementDtype) {
      throw Exception(
          "Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}");
    }

    if (this.size() == 0) {
      throw Exception('Trying to pop from an empty list.');
    }
    final outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    final tensor = this.tensors.removeLast();

    assertShapesMatchAllowUndefinedSize(
        tensor.shape, elementShape, 'TensorList shape mismatch: ');

    return reshape(tensor, outputElementShape);
  }

  /**
   * Push a tensor to the end of the list.
   * @param tensor Tensor to be pushed.
   */
  pushBack(Tensor tensor) {
    if (tensor.dtype != this.elementDtype) {
      throw Exception(
          "Invalid data types; op elements ${tensor.dtype}, but list elements ${this.elementDtype}");
    }

    assertShapesMatchAllowUndefinedSize(
        tensor.shape, this.elementShape, 'TensorList shape mismatch: ');

    if (this.maxNumElements == this.size()) {
      throw Exception("Trying to push element into a full list.");
    }
    keep(tensor);
    this.tensors.add(tensor);
  }

  /**
   * Update the size of the list.
   * @param size the new size of the list.
   */
  resize(int size) {
    if (size < 0) {
      throw Exception(
          "TensorListResize expects size to be non-negative. Got: ${size}");
    }

    if (this.maxNumElements != -1 && size > this.maxNumElements) {
      throw Exception(
          "TensorListResize input size ${size} is greater maxNumElement ${this.maxNumElements}.");
    }
    this.tensors.length = size;
  }

  /**
   * Retrieve the element at the provided index
   * @param elementShape shape of the tensor
   * @param elementDtype dtype of the tensor
   * @param elementIndex index of the tensor
   */
  Tensor getItem(
      int elementIndex, List<int> elementShape, DataType elementDtype) {
    if (elementDtype != this.elementDtype) {
      throw Exception(
          "Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}");
    }
    if (elementIndex < 0 || elementIndex > this.tensors.length) {
      throw Exception(
          "Trying to access element ${elementIndex} in a list with ${this.tensors.length} elements.");
    }

    if (this.tensors[elementIndex] == null) {
      throw Exception("element at index ${elementIndex} is null.");
    }

    assertShapesMatchAllowUndefinedSize(this.tensors[elementIndex].shape,
        elementShape, 'TensorList shape mismatch: ');
    final outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    return reshape(this.tensors[elementIndex], outputElementShape);
  }

  /**
   * Set the tensor at the index
   * @param elementIndex index of the tensor
   * @param tensor the tensor to be inserted into the list
   */
  setItem(int elementIndex, Tensor tensor) {
    if (tensor.dtype != this.elementDtype) {
      throw Exception(
          "Invalid data types; op elements ${tensor.dtype}, but list elements ${this.elementDtype}");
    }

    if (elementIndex < 0 ||
        this.maxNumElements != -1 && elementIndex >= this.maxNumElements) {
      throw Exception(
          "Trying to set element ${elementIndex} in a list with max ${this.maxNumElements} elements.");
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensor.shape, 'TensorList shape mismatch: ');
    keep(tensor);
    this.tensors[elementIndex] = tensor;
  }

  /**
   * Return selected values in the TensorList as a stacked Tensor. All of
   * selected values must have been written and their shapes must all match.
   * @param indices indices of tensors to gather
   * @param elementDtype output tensor dtype
   * @param elementShape output tensor element shape
   */
  Tensor gather(
      List<int> indices, DataType elementDtype, List<int> elementShape) {
    if (elementDtype != this.elementDtype) {
      throw Exception(
          "Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}");
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, elementShape, 'TensorList shape mismatch: ');

    // When indices is greater than the size of the list, indices beyond the
    // size of the list are ignored.
    indices = indices.slice(0, this.size());
    final outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    if (indices.length == 0) {
      return tensor([], [0, ...outputElementShape]);
    }

    return tidy(() {
      final tensors = indices
          .map((i) => reshape(this.tensors[i], outputElementShape))
          .toList();
      return tf.stack(tensors, 0);
    });
  }

  /**
   * Return the values in the TensorList as a concatenated Tensor.
   * @param elementDtype output tensor dtype
   * @param elementShape output tensor element shape
   */
  Tensor concat(DataType elementDtype, List<int> elementShape) {
    if (elementDtype != null && elementDtype != this.elementDtype) {
      throw Exception(
          "TensorList dtype is ${this.elementDtype} but concat requested dtype ${elementDtype}");
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, elementShape, 'TensorList shape mismatch: ');
    final outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);

    if (this.size() == 0) {
      return tensor([], [0, ...outputElementShape]);
    }
    return tidy(() {
      final tensors =
          this.tensors.map((t) => reshape(t, outputElementShape)).toList();
      return tf.concat(tensors, 0);
    });
  }
}

/**
 * Creates a TensorList which, when stacked, has the value of tensor.
 * @param tensor from tensor
 * @param elementShape output tensor element shape
 */
TensorListContainer fromTensor(
  Tensor tensor,
  List<int> elementShape,
  DataType elementDtype,
) {
  final dtype = tensor.dtype;
  if (tensor.shape.length < 1) {
    throw Exception(
        "Tensor must be at least a vector, but saw shape: ${tensor.shape}");
  }
  if (tensor.dtype != elementDtype) {
    throw Exception(
        "Invalid data types; op elements ${tensor.dtype}, but list elements ${elementDtype}");
  }
  final tensorElementShape = tensor.shape.slice(1);
  assertShapesMatchAllowUndefinedSize(
      tensorElementShape, elementShape, 'TensorList shape mismatch: ');
  final List<Tensor> tensorList = unstack(tensor);
  return TensorListContainer(tensorList, elementShape, dtype);
}

/**
 * Return a TensorList of the given size with empty elements.
 * @param elementShape the shape of the future elements of the list
 * @param elementDtype the desired type of elements in the list
 * @param numElements the number of elements to reserve
 */
TensorListContainer reserve(
    List<int> elementShape, DataType elementDtype, int numElements) {
  return TensorListContainer([], elementShape, elementDtype, numElements);
}

/**
 * Put tensors at specific indices of a stacked tensor into a TensorList.
 * @param indices list of indices on how to scatter the tensor.
 * @param tensor input tensor.
 * @param elementShape the shape of the future elements of the list
 * @param numElements the number of elements to scatter
 */
TensorListContainer scatter(
  Tensor tensor,
  List<int> indices,
  List<int> elementShape,
  int? numElements,
) {
  if (indices.length != tensor.shape[0]) {
    throw Exception(
        "Expected len(indices) == tensor.shape[0], but saw: ${indices.length} vs. ${tensor.shape[0]}");
  }

  final maxIndex = indices.reduce(math.max);

  if (numElements != null && numElements != -1 && maxIndex >= numElements) {
    throw Exception(
        "Max index must be < array size (${maxIndex}  vs. ${numElements})");
  }

  final list = TensorListContainer([], elementShape, tensor.dtype, numElements);
  final tensors = unstack(tensor, 0);
  int index = 0;
  indices.forEach((value) {
    list.setItem(value, tensors[index++]);
  });
  return list;
}

/**
 * Split the values of a Tensor into a TensorList.
 * @param length the lengths to use when splitting value along
 *    its first dimension.
 * @param tensor the tensor to split.
 * @param elementShape the shape of the future elements of the list
 */
TensorListContainer split(
    Tensor tensor, List<int> length, List<int> elementShape) {
  int totalLength = 0;
  final cumulativeLengths = length.map((len) {
    totalLength += len;
    return totalLength;
  }).toList();

  if (totalLength != tensor.shape[0]) {
    throw Exception("Expected sum of lengths to be equal to tensor.shape[0],"
        " but sum of lengths is ${totalLength}, and tensor's shape is: ${tensor.shape}");
  }

  final shapeWithoutFirstDim = tensor.shape.slice(1);
  final outputElementShape =
      mergeElementShape(shapeWithoutFirstDim, elementShape);
  final elementPerRow = totalLength == 0 ? 0 : tensor.size ~/ totalLength;
  final List<Tensor> tensors = tidy(() {
    final List<Tensor> tensors = [];
    tensor = reshape(tensor, [1, totalLength, elementPerRow]);
    for (int i = 0; i < length.length; ++i) {
      final previousLength = (i == 0) ? 0 : cumulativeLengths[i - 1];
      final indices = [0, previousLength, 0];
      final sizes = [1, length[i], elementPerRow];
      tensors[i] = reshape(
          slice(tensor, indices, sizes), outputElementShape as List<int>);
    }
    tensor.dispose();
    return tensors;
  });

  final list =
      TensorListContainer([], elementShape, tensor.dtype, length.length);

  for (int i = 0; i < tensors.length; i++) {
    list.setItem(i, tensors[i]);
  }
  return list;
}
