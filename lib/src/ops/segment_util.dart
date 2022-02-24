import 'dart:math' as math;
import 'package:tensorflow_wasm/src/ops/reduce_util.dart'
    show PARALLELIZE_THRESHOLD;
import 'package:tensorflow_wasm/src/tensor.dart';

import '../util_base.dart' show nearestDivisor;

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

// import {TensorInfo} from '../kernel_registry';
// import {nearestDivisor} from '../util';

// import {PARALLELIZE_THRESHOLD} from './reduce_util';

class SegOpInfo {
  final int windowSize;
  final int batchSize;
  final int inSize;
  final int numSegments;

  SegOpInfo({
    required this.windowSize,
    required this.batchSize,
    required this.inSize,
    required this.numSegments,
  });
}

int segOpComputeOptimalWindowSize(int inSize, int numSegments) {
  bool done = false;
  int res;

  if (inSize <= PARALLELIZE_THRESHOLD) {
    res = inSize;
    done = true;
  } else {
    res = nearestDivisor(inSize, math.sqrt(inSize).floor());
  }

  while (!done) {
    if (res > numSegments || res == inSize) {
      done = true;
    } else {
      res = nearestDivisor(inSize, res + 1);
    }
  }
  return res;
}

Shape computeSegmentOutShape(Shape aShape, int axis, int numSegments) {
  final Shape outShape = [];
  final rank = aShape.length;
  for (int dim = 0; dim < rank; dim++) {
    if (dim != axis) {
      outShape.add(aShape[dim]);
    } else {
      outShape.add(numSegments);
    }
  }
  return outShape;
}

class GatherOpShapeInfo {
  final int batchSize;
  final int sliceSize;
  final int outerSize;
  final int dimSize;
  final List<int> outputShape;

  GatherOpShapeInfo({
    required this.batchSize,
    required this.sliceSize,
    required this.outerSize,
    required this.dimSize,
    required this.outputShape,
  });
}

GatherOpShapeInfo collectGatherOpShapeInfo(
    TensorInfo x, TensorInfo indices, int axis, int batchDims) {
  final indicesRank = indices.shape.length;
  final xRank = x.shape.length;

  if (batchDims != 0) {
    if (batchDims < -indicesRank || batchDims > indicesRank) {
      throw Exception(
          'Expect batchDims in the range of [-${indicesRank}, ${indicesRank}], but got ${batchDims}');
    }
  }

  if (batchDims < 0) {
    batchDims += indicesRank;
  }

  if (batchDims > xRank) {
    throw Exception(
        'batchDims (${batchDims}) must be less than rank(x) (${xRank}).');
  }

  if (axis < batchDims) {
    throw Exception(
        'batchDims (${batchDims}) must be less than or equal to axis (${axis}).');
  }

  for (int i = 0; i < batchDims; ++i) {
    if (x.shape[i] != indices.shape[i]) {
      throw Exception(
          'x.shape[${i}]: ${x.shape[i]} should be equal to indices.shape[${i}]: ${indices.shape[i]}.');
    }
  }
  final dimSize = x.shape[axis];

  final Shape outputShape = [];
  int batchSize = 1;
  int outerSize = 1;
  int sliceSize = 1;

  for (int i = 0; i < batchDims; ++i) {
    outputShape.add(x.shape[i]);
    batchSize *= x.shape[i];
  }

  for (int i = batchDims; i < axis; i++) {
    outputShape.add(x.shape[i]);
    outerSize *= x.shape[i];
  }

  for (int i = batchDims; i < indicesRank; i++) {
    outputShape.add(indices.shape[i]);
  }

  for (int i = axis + 1; i < xRank; i++) {
    outputShape.add(x.shape[i]);
    sliceSize *= x.shape[i];
  }

  return GatherOpShapeInfo(
    batchSize: batchSize,
    sliceSize: sliceSize,
    outerSize: outerSize,
    dimSize: dimSize,
    outputShape: outputShape,
  );
}
