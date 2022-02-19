/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

// import * as util from '../util';
import 'package:tensorflow_wasm/src/tensor.dart';

import '../util_base.dart' as util;
import 'package:collection/collection.dart';

/**
 * Returns true if the axis specifies the inner most dimensions of the
 * array.
 */
bool axesAreInnerMostDims(Shape axes, int rank) {
  for (int i = 0; i < axes.length; ++i) {
    if (axes[axes.length - i - 1] != rank - 1 - i) {
      return false;
    }
  }
  return true;
}

Shape combineLocations(Shape outputLoc, Shape reduceLoc, Shape axes) {
  final rank = outputLoc.length + reduceLoc.length;
  final Shape loc = [];
  int outIdx = 0;
  int reduceIdx = 0;
  for (int dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) == -1) {
      loc.add(outputLoc[outIdx++]);
    } else {
      loc.add(reduceLoc[reduceIdx++]);
    }
  }
  return loc;
}

class OutAndReduceShapes {
  final Shape outShape;
  final Shape reduceShape;
  OutAndReduceShapes({
    required this.outShape,
    required this.reduceShape,
  });
}

OutAndReduceShapes computeOutAndReduceShapes(Shape aShape, Shape axes) {
  final Shape outShape = [];
  final rank = aShape.length;
  for (int dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) == -1) {
      outShape.add(aShape[dim]);
    }
  }
  final reduceShape = axes.map((dim) => aShape[dim]).toList();
  return OutAndReduceShapes(outShape: outShape, reduceShape: reduceShape);
}

Shape expandShapeToKeepDim(Shape shape, Shape axes) {
  final reduceSubShape = axes.map((x) => 1).toList();
  return combineLocations(shape, reduceSubShape, axes);
}

void assertAxesAreInnerMostDims(String msg, Shape axes, int rank) {
  util.assert_(
      axesAreInnerMostDims(axes, rank),
      () =>
          '${msg} supports only inner-most axes for now. ' +
          'Got axes ${axes} and rank-${rank} input.');
}

/**
 * Returns the axes permutation to be used with `tf.transpose`, if such
 * permutation is necessary. Otherwise it returns null. This method is used by
 * operations that operate only on inner-most axes.
 */
List<int>? getAxesPermutation(Shape axes, int rank) {
  if (axesAreInnerMostDims(axes, rank)) {
    return null;
  }
  final List<int> result = [];
  for (int i = 0; i < rank; ++i) {
    if (axes.indexOf(i) == -1) {
      result.add(i);
    }
  }
  axes.forEach((axis) => result.add(axis));
  return result;
}

/** Returns the axes permutation that undoes the original permutation. */
List<int> getUndoAxesPermutation(List<int> axes) {
  return (axes.mapIndexed((i, axis) => [i, axis]).toList()
        ..sort((a, b) => a[1] - b[1]))
      .map((x) => x[0])
      .toList();
}

List<int> getInnerMostAxes(int numAxes, int rank) {
  final List<int> res = [];
  for (int i = rank - numAxes; i < rank; ++i) {
    res.add(i);
  }
  return res;
}
