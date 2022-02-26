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

/**
 * Gets the new shape of the input Tensor after it's been reshaped
 * to:
 * [blockShape[0], ..., blockShape[M-1], batch / prod(blockShape),
 * inputShape[1], ..., inputShape[N-1]]
 *
 * See step 1: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */

import 'package:tensorflow_wasm/tensorflow_wasm.dart' show SliceList;

List<int> getReshaped(
  List<int> inputShape,
  List<int> blockShape,
  int prod, {
  bool batchToSpace = true,
}) {
  List<int> reshaped = [];
  if (batchToSpace) {
    reshaped = [...reshaped, ...blockShape.sublistRelaxed(0)];
    reshaped.add(inputShape[0] ~/ prod);
    reshaped = [...reshaped, ...inputShape.sublistRelaxed(1)];
  } else {
    reshaped = [...reshaped, inputShape[0]];
    final spatialLength = blockShape.length;
    for (int i = 0; i < spatialLength; ++i) {
      reshaped = [
        ...reshaped,
        inputShape[i + 1] ~/ blockShape[i],
        blockShape[i]
      ];
    }
    reshaped = [...reshaped, ...inputShape.sublistRelaxed(spatialLength + 1)];
  }
  return reshaped;
}

/**
 * Gets the permutation that will transpose the dimensions of the
 * reshaped tensor to shape:
 *
 * [batch / prod(block_shape),inputShape[1], blockShape[0], ...,
 * inputShape[M], blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
 *
 * see step 2: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */
List<int> getPermuted(
  int reshapedRank,
  int blockShapeRank, {
  bool batchToSpace = true,
}) {
  final List<int> permuted = [];
  if (batchToSpace) {
    permuted.add(blockShapeRank);
    for (int i = blockShapeRank + 1; i < reshapedRank; ++i) {
      if (i <= 2 * blockShapeRank) {
        permuted.add(i);
        permuted.add(i - (blockShapeRank + 1));
      } else {
        permuted.add(i);
      }
    }
  } else {
    final List<int> permutedBeforeBatch = [];
    final List<int> permutedAfterBatch = [];
    for (int i = 1; i < reshapedRank; ++i) {
      if (i >= blockShapeRank * 2 + 1 || i % 2 == 1) {
        permutedAfterBatch.add(i);
      } else {
        permutedBeforeBatch.add(i);
      }
    }
    permuted.addAll(permutedBeforeBatch);
    permuted.add(0);
    permuted.addAll(permutedAfterBatch);
  }
  return permuted;
}

/**
 * Gets the shape of the reshaped and permuted input Tensor before any cropping
 * is applied.  The new shape will be:
 *
 * [batch / prod(blockShape),inputShape[1] * blockShape[0], ...,
 * inputShape[M] * blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
 *
 * See step 3: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */
List<int> getReshapedPermuted(
  List<int> inputShape,
  List<int> blockShape,
  int prod, {
  bool batchToSpace = true,
}) {
  final reshapedPermuted = <int>[];

  if (batchToSpace) {
    reshapedPermuted.add(inputShape[0] ~/ prod);
  } else {
    reshapedPermuted.add(inputShape[0] * prod);
  }

  for (int i = 1; i < inputShape.length; ++i) {
    if (i <= blockShape.length) {
      if (batchToSpace) {
        reshapedPermuted.add(blockShape[i - 1] * inputShape[i]);
      } else {
        reshapedPermuted.add(inputShape[i] ~/ blockShape[i - 1]);
      }
    } else {
      reshapedPermuted.add(inputShape[i]);
    }
  }

  return reshapedPermuted;
}

/**
 * Converts the crops argument into the beginning coordinates of a slice
 * operation.
 */
List<int> getSliceBeginCoords(List<List<int>> crops, int blockShape) {
  final sliceBeginCoords = [
    ...[0]
  ];
  for (int i = 0; i < blockShape; ++i) {
    sliceBeginCoords.add(crops[i][0]);
  }
  return sliceBeginCoords;
}

/**
 * Converts the crops argument into the size of a slice operation.  When
 * combined with getSliceBeginCoords this function allows the reshaped and
 * permuted Tensor to be cropped to its final output shape of:
 *
 * inputShape[1] * blockShape[0] - crops[0,0] - crops[0,1], ...,
 * inputShape[M] * blockShape[M-1] -crops[M-1,0] -
 * crops[M-1,1],inputShape[M+1], ..., inputShape[N-1]]
 *
 * See step 4: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */
List<int> getSliceSize(
    List<int> uncroppedShape, List<List<int>> crops, int blockShape) {
  final sliceSize = uncroppedShape.sublistRelaxed(0, 1);
  for (int i = 0; i < blockShape; ++i) {
    sliceSize.add(uncroppedShape[i + 1] - crops[i][0] - crops[i][1]);
  }

  return sliceSize;
}
