/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
// import * as util from '../util';

import 'dart:math' as math;
import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/src/tensor.dart';

import '../util_base.dart' as util;

const NEW_AXIS = -2;
const SHRINK_AXIS = -1;
const MAX_SAFE_INTEGER = 9007199254740991;
const MIN_SAFE_INTEGER = -9007199254740991;

// Sparse slicing specification
// if one does foo[3:5, ..., -3], the begin, end and strides will have length
// of 3.
class _StridedSliceSparseSpec {
  int dims;
  int numAddAxisAfterEllipsis;
  final List<int> begin;
  final List<int> end;
  final List<int> strides;
  final int beginMask;
  final int endMask;
  int ellipsisMask;
  final int newAxisMask;
  final int shrinkAxisMask;

  _StridedSliceSparseSpec({
    required this.dims,
    required this.numAddAxisAfterEllipsis,
    required this.begin,
    required this.end,
    required this.strides,
    required this.beginMask,
    required this.endMask,
    required this.ellipsisMask,
    required this.newAxisMask,
    required this.shrinkAxisMask,
  });
}

// Dense slicing specification
// all ellipses and newaxis are expanded out. So if foo[3:5, ..., -3] where foo
// is 10 dimensional, each array of begin, end, strides will have 10 entries
// where as the sparse can have length less than the rank of foo.
class _StridedSliceDenseSpec {
  final int dims;
  int beginMask = 0;
  int endMask = 0;
  bool beginValid;
  bool endValid;
  late final List<int> begin = List.filled(dims, 0);
  late final List<int> end = List.filled(dims, 0);
  late final List<int> strides = List.filled(dims, 0);
  // This array helps construct the final shape of the slice.
  // The final tensor is reduced in rank whenever a single index e.g. foo[3]
  // is called for. The final tensor increases in rank with newAxis entries.
  // If an index in this array is positive, the size of the dimension is
  // obtained from canonical end-begin.  Otherwise, if it is a NEW_AXIS, it will
  // be 1. A shrunk dimension is skipped.
  final List<int> finalShapeGatherIndices = [];
  // This array has the same size as finalShapeGatherIndices, but it remembers
  // the sparse index that a dimension comes from, instead of dense index.
  // A -1 in this vector means the index is not from the sparse input.
  final List<int> finalShapeGatherIndicesSparse = [];
  late final List<int> inputShapeGatherIndicesSparse = List.filled(dims, 0);
  // The dense indexed shrink mask is which processing dimensions should be
  // shrunk. For example, if foo.shape = [10, 10, 10, 10], foo[3, ..., 5] has
  // sparseShrinkAxisMask of 5 (0101) and denseShrinkAxisMask of 9 (1001),
  // yielding a final shape [10, 10].
  int shrinkAxisMask = 0;

  // dense.begin = new Array(dense.dims);
  // dense.end = new Array(dense.dims);
  // dense.strides = new Array(dense.dims);
  // dense.finalShapeGatherIndices = [];
  // dense.finalShapeGatherIndicesSparse = [];
  // dense.inputShapeGatherIndicesSparse = new Array(dense.dims);

  _StridedSliceDenseSpec({
    required this.dims,
    required this.beginValid,
    required this.endValid,
  });
}

class SliceInfo {
  final List<int> finalShapeSparse;
  final List<int> finalShape;
  final bool isIdentity;
  final bool sliceDim0;
  final bool isSimpleSlice;
  final List<int> begin;
  final List<int> end;
  final List<int> strides;

  SliceInfo({
    required this.finalShapeSparse,
    required this.finalShape,
    required this.isIdentity,
    required this.sliceDim0,
    required this.isSimpleSlice,
    required this.begin,
    required this.end,
    required this.strides,
  });
}

void assertParamsValid(TensorInfo input, List<int> begin, List<int> size) {
  final inputRank = input.shape.length;
  util.assert_(
      inputRank == begin.length,
      () =>
          "Error in slice${inputRank}D: Length of begin ${begin} must " +
          "match the rank of the array (${inputRank}).");
  util.assert_(
      inputRank == size.length,
      () =>
          "Error in slice${inputRank}D: Length of size ${size} must " +
          "match the rank of the array (${inputRank}).");

  for (int i = 0; i < inputRank; ++i) {
    util.assert_(
        begin[i] + size[i] <= input.shape[i],
        () =>
            "Error in slice${inputRank}D: begin[${i}] + size[${i}] " +
            "(${begin[i] + size[i]}) would overflow input.shape[${i}] (${input.shape[i]})");
  }
}

/** Converts a binary mask to an array of axes. Used in stridedSlice(). */
List<int> maskToAxes(int mask) {
  final axes = <int>[];
  int axis = 0;
  while (mask > 0) {
    if (mask & 1 != 0) {
      axes.add(axis);
    }
    mask ~/= 2;
    axis++;
  }
  return axes;
}

bool _intBool(int value) => value != 0;

/** Computes the output shape given the strided slice params. */
List<int> computeSliceOutShape(List<int> begin, List<int> end, List<int> strides) {
  final size = <int>[];
  for (int axis = 0; axis < begin.length; axis++) {
    size[axis] = ((end[axis] - begin[axis]) / strides[axis]).ceil();
  }
  return size;
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stride value. Otherwise, insert.
List<int> stridesWithElidedDims(List<int> strides, int ellipsisInsertionIndex,
    int numElidedAxes, List<int> inputShape) {
  final newStrides = [...strides];
  for (int i = newStrides.length; i < inputShape.length; i++) {
    newStrides.add(1);
  }
  for (int i = 0; i < numElidedAxes; i++) {
    if (i == 0) {
      newStrides[ellipsisInsertionIndex] = 1;
    } else {
      newStrides.insert(
        ellipsisInsertionIndex,
        1 /* element to add */,
      );
      newStrides.removeLast();
    }
  }
  return newStrides;
}

int _unnormalizeAxis(
    int ellipsisInsertionIndex, int numElidedAxes, int normalizedAxis) {
  if (normalizedAxis <= ellipsisInsertionIndex) {
    return normalizedAxis;
  }

  return normalizedAxis - (numElidedAxes - 1);
}

List<int> _getElidedAxes(int numElidedAxes, int ellipsisInsertionIndex) {
  final elidedAxes = <int>[];
  for (int i = 0; i < numElidedAxes; i++) {
    elidedAxes.add(ellipsisInsertionIndex + i);
  }
  return elidedAxes;
}

class NormalizedAxes {
  final List<int> begin;
  final List<int> end;
  final List<int> strides;

  NormalizedAxes({
    required this.begin,
    required this.end,
    required this.strides,
  });
}

// Normalize the start, end and strides.
NormalizedAxes getNormalizedAxes(
    List<int> inputShape,
    List<int> ellipsisAxes,
    int numInterpolatedAxes,
    List<int> begin,
    List<int> end,
    List<int> strides,
    int beginMask,
    int endMask,
    int ellipsisMask) {
  final inputRank = inputShape.length;
  var normalizedBegin = List.filled(inputRank, 0),
      normalizedEnd = List.filled(inputRank, 0),
      normalizedStrides = List.filled(inputRank, 0);
  if (ellipsisAxes.isNotEmpty && numInterpolatedAxes > 0) {
    final fullIndex = ellipsisAxes[0];

    // The ellipsis applies to the masked index as well as any dimensions
    // that are interpolated.
    final numElidedAxes = numInterpolatedAxes + 1;
    normalizedBegin = startIndicesWithElidedDims(
        beginMask, fullIndex, numElidedAxes, begin, inputShape);
    normalizedEnd = stopIndicesWithElidedDims(
        endMask, fullIndex, numElidedAxes, end, inputShape);
    normalizedStrides =
        stridesWithElidedDims(strides, fullIndex, numElidedAxes, inputShape);
  } else {
    for (int axis = 0; axis < inputRank; axis++) {
      normalizedBegin[axis] = startForAxis(
          beginMask, begin, strides, inputShape, axis, ellipsisMask);
      normalizedEnd[axis] =
          stopForAxis(endMask, end, strides, inputShape, axis, ellipsisMask);
      normalizedStrides[axis] = stridesForAxis(strides, axis, ellipsisMask);
    }
  }

  return NormalizedAxes(
      begin: normalizedBegin, end: normalizedEnd, strides: normalizedStrides);
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current start value. Otherwise, insert.
List<int> startIndicesWithElidedDims(int beginMask, int ellipsisInsertionIndex,
    int numElidedAxes, List<int> originalBegin, List<int> inputShape) {
  final newIndices = [...inputShape];
  final elidedAxes = _getElidedAxes(numElidedAxes, ellipsisInsertionIndex);

  for (int axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = 0;
    } else {
      final originalAxis =
          _unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
      int originalValue = originalBegin[originalAxis];
      if (_intBool(beginMask & 1 << originalAxis)) {
        originalValue = 0;
      }

      newIndices[axis] = originalValue;
    }
  }
  return newIndices;
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stop value. Otherwise, insert.
List<int> stopIndicesWithElidedDims(int endMask, int ellipsisInsertionIndex,
    int numElidedAxes, List<int> originalEnd, List<int> inputShape) {
  final newIndices = [...inputShape];
  final elidedAxes = _getElidedAxes(numElidedAxes, ellipsisInsertionIndex);

  for (int axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = MAX_SAFE_INTEGER;
    } else {
      final originalAxis =
          _unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
      int originalValue = originalEnd[originalAxis];
      if (_intBool(endMask & 1 << originalAxis)) {
        originalValue = MAX_SAFE_INTEGER;
      }
      newIndices[axis] = originalValue;
    }
  }

  for (int i = 0; i < newIndices.length; i++) {
    // Handle negative indices
    final axisSize = inputShape[i];
    if (newIndices[i] < 0) {
      newIndices[i] += axisSize;
    }
    newIndices[i] = util.clamp(0, newIndices[i], inputShape[i]);
  }
  return newIndices;
}

int stridesForAxis(List<int> strides, int axis, int ellipsisMask) {
  int stride = strides[axis];
  if (_intBool(ellipsisMask & (1 << axis)) || stride == null) {
    stride = 1;
  }

  return stride;
}

int startForAxis(int beginMask, List<int> startIndices, List<int> strides,
    List<int> inputShape, int axis, int ellipsisMask) {
  // Begin with the specified index
  int start = startIndices[axis];
  final stride = axis < strides.length ? strides[axis] : 1;

  // Check the axis bit from right of masked axes, or the begin index is not set
  // for the axis.
  if (_intBool(beginMask & 1 << axis) ||
      _intBool(ellipsisMask & 1 << axis) ||
      start == null) {
    if (stride > 0) {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = MIN_SAFE_INTEGER;
    } else {
      // Backward iteration - use the last element.
      start = MAX_SAFE_INTEGER;
    }
  }

  // Handle negative indices
  final axisSize = inputShape[axis];
  if (start < 0) {
    start += axisSize;
  }

  // Clamping
  start = util.clamp(0, start, axisSize - 1);

  return start;
}

int stopForAxis(int endMask, List<int> stopIndices, List<int> strides,
    List<int> inputShape, int axis, int ellipsisMask) {
  // Begin with the specified index
  int? stop = axis < stopIndices.length ? stopIndices[axis] : null;
  final stride = axis < strides.length ? strides[axis] : 1;

  // Check the axis bit from right of masked axes, or if the stop index is not
  // set for this axis.
  if (_intBool(endMask & (1 << axis)) ||
      _intBool(ellipsisMask & (1 << axis)) ||
      stop == null) {
    if (stride > 0) {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = MAX_SAFE_INTEGER;
    } else {
      // Backward iteration - use the first element.
      stop = MIN_SAFE_INTEGER;
    }
  }

  // Handle negative indices
  final axisSize = inputShape[axis];
  if (stop! < 0) {
    stop += axisSize;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (stride > 0) {
    // Forward iteration
    stop = util.clamp(0, stop, axisSize);
  } else {
    // Backward iteration
    stop = util.clamp(-1, stop, axisSize - 1);
  }

  return stop;
}

/**
 * Returns true if the slice occupies a continous set of elements in the
 * 'flat' space.
 */
bool isSliceContinous(List<int> shape, List<int> begin, List<int> size) {
  // Index of the first axis that has size > 1.
  int firstNonOneAxis = size.length;
  for (int i = 0; i < size.length; i++) {
    if (size[i] > 1) {
      firstNonOneAxis = i;
      break;
    }
  }

  for (int i = firstNonOneAxis + 1; i < size.length; i++) {
    if (begin[i] > 0 || size[i] != shape[i]) {
      return false;
    }
  }
  return true;
}

int computeFlatOffset(List<int> begin, List<int> strides) {
  int flatOffset = begin.length > 0 ? begin[begin.length - 1] : 1;
  for (int i = 0; i < begin.length - 1; i++) {
    flatOffset += begin[i] * strides[i];
  }
  return flatOffset;
}

List<List<int>> parseSliceParams(
    TensorInfo x, List<int> begin, List<int>? size) {
  // The following logic allows for more ergonomic calls.
  List<int> begin_;
  final xRank = x.shape.length;
  if (begin is int) {
    begin_ = [begin as int, ...List.filled(xRank - 1, 0)];
  } else if (begin.length < xRank) {
    begin_ = [...begin, ...Iterable.generate(xRank - begin.length, (_) => 0)];
  } else {
    begin_ = [...begin];
  }
  begin_.forEach((d) {
    util.assert_(
        d != -1, () => 'slice() does not support negative begin indexing.');
  });
  List<int> size_;
  if (size == null) {
    size_ = List.filled(xRank, -1);
  } else if (size is int) {
    size_ = [size as int, ...Iterable.generate(xRank - 1, (_) => -1)];
  } else if (size.length < xRank) {
    size_ = [...size, ...Iterable.generate(xRank - size.length, (_) => -1)];
  } else {
    size_ = size;
  }
  size_ = size_.mapIndexed((i, d) {
    if (d >= 0) {
      return d;
    } else {
      util.assert_(
          d == -1,
          () =>
              "Negative size values should be exactly -1 but got " +
              "${d} for the slice() size at index ${i}.");
      return x.shape[i] - begin_[i];
    }
  }).toList();
  return [begin_, size_];
}

// Convert the slicing specification from a sparse representation to a dense
// representation. This means that all ellipses and newaxis are expanded out.
SliceInfo sliceInfo(
    List<int> xShape,
    List<int> begin,
    List<int> end,
    List<int> strides,
    int beginMask,
    int endMask,
    int ellipsisMask,
    int newAxisMask,
    int shrinkAxisMask) {
  final List<int> stridesNonNull;
  if (strides == null) {
    stridesNonNull = List.filled(begin.length, 1);
  } else {
    stridesNonNull = strides;
  }

  // Only one non-zero bit is allowed in ellipsisMask, which means ellipsisMask
  // is a power of 2. Use bit compares to ensure ellipsisMask is 0 or a power
  // of 2. When i is a power of 2, i & (i - 1) is always 0.
  // Also ref:
  // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
  if (ellipsisMask != null && (ellipsisMask & (ellipsisMask - 1)) != 0) {
    throw Exception('Multiple ellipses in slice is not allowed.');
  }

  // Step 1: Account for ellipsis and new axis.
  // Check for ellipsis and count how many non-newaxis there are after.
  bool ellipsisSeen = false;

  final sparseSpec = _StridedSliceSparseSpec(
      dims: stridesNonNull.length,
      numAddAxisAfterEllipsis: 0,
      begin: [...begin],
      end: [...end],
      strides: [...stridesNonNull],
      beginMask: beginMask,
      endMask: endMask,
      ellipsisMask: ellipsisMask,
      newAxisMask: newAxisMask,
      shrinkAxisMask: shrinkAxisMask);

  for (int i = 0; i < sparseSpec.dims; i++) {
    if (ellipsisSeen && ((1 << i) & newAxisMask) != 0) {
      sparseSpec.numAddAxisAfterEllipsis++;
    }
    if (_intBool((1 << i) & ellipsisMask)) {
      ellipsisSeen = true;
    }
  }
  // If no ellipsis insert one at the end.
  if (!ellipsisSeen) {
    sparseSpec.ellipsisMask |= (1 << sparseSpec.dims);
    sparseSpec.dims++; // this effects loop iteration below
  }

  // Step 2: Make a sparse spec into a full index spec.
  //
  // The sparse spec deos not correspond to the number of dimensions.
  // Make a dense spec that cooresponds to the number of dimensions.
  //
  // For example suppose foo[...,3:] on foo.shape = [2, 2, 3] then we need to
  // produce the missing beginMask for the first two dimensions i.e. from
  // beginMaskSpec = 0, endMaskSpec = 2, we achieve beginMask = 6 (110),
  // endMask = 7 (111).
  final denseSpec = _StridedSliceDenseSpec(
    dims: xShape.length,
    beginValid: false,
    endValid: false,
  );

  _buildDenseSpec(sparseSpec, denseSpec);

  // Step 3: Make implicit ranges (non-zero beginMasks and endMasks) explicit
  // and bounds check.
  bool isIdentity = true;
  bool sliceDim0 = true;
  bool isSimpleSlice = true;
  final processingShape = <int>[];
  final finalShape = <int>[];

  for (int i = 0; i < xShape.length; ++i) {
    if (denseSpec.strides[i] == 0) {
      throw Exception("strides[${i}] must be non-zero");
    }
    final shrinkI = !!_intBool(denseSpec.shrinkAxisMask & (1 << i));
    final dimI = xShape[i];
    if (dimI == -1) {
      processingShape.add(shrinkI ? 1 : -1);
      continue;
    }

    final masks = [
      denseSpec.beginMask & (1 << i),
      denseSpec.endMask & (1 << i)
    ];
    final validRange = [
      denseSpec.strides[i] > 0 ? 0 : -1,
      denseSpec.strides[i] > 0 ? dimI : dimI - 1
    ];

    if (shrinkI && denseSpec.strides[i] <= 0) {
      throw Exception('only stride 1 allowed on non-range indexing.');
    }

    isSimpleSlice = isSimpleSlice && (denseSpec.strides[i] == 1);

    final beginAndEndMasked = !!(_intBool(denseSpec.beginMask & (1 << i)) &&
        _intBool(denseSpec.endMask & (1 << i)));

    if (denseSpec.beginValid && denseSpec.endValid) {
      if (shrinkI) {
        // If we are shrinking, the end index is now possibly incorrect. In
        // particular foo[-1] produces sparseBegin = -1, sparseEnd = 0.
        // and canonical puts these to n-1 and 0, which implies a degenerate
        // interval. Fortunately, it is now safe to re-create end as begin + 1.
        final xFwd = denseSpec.begin[i] < 0
            ? dimI + denseSpec.begin[i]
            : denseSpec.begin[i];
        denseSpec.begin[i] = xFwd;
        denseSpec.end[i] = denseSpec.begin[i] + 1;
        if (xFwd < 0 || xFwd >= dimI) {
          throw Exception(
              "slice index ${denseSpec.begin[i]} of dimension ${i} out of bounds.");
        }
      } else {
        denseSpec.begin[i] = _canonical(denseSpec.begin[i], 0,
            denseSpec.strides[i], dimI, masks, validRange);
        denseSpec.end[i] = _canonical(
            denseSpec.end[i], 1, denseSpec.strides[i], dimI, masks, validRange);
      }
      // Update optimization values
      final takeAllInDimension = denseSpec.strides[i] == 1 &&
          denseSpec.begin[i] == 0 &&
          denseSpec.end[i] == dimI;
      isIdentity = isIdentity && takeAllInDimension;
      sliceDim0 = sliceDim0 &&
          ((i == 0 && denseSpec.strides[i] == 1) || takeAllInDimension);
    } else {
      isIdentity =
          isIdentity && ((denseSpec.strides[i] == 1) && beginAndEndMasked);
      sliceDim0 = sliceDim0 &&
          ((i == 0 && denseSpec.strides[i] == 1) || beginAndEndMasked);
    }
    // Compute the processing shape (the intermediate Eigen will produce)
    int? intervalLength;
    if (denseSpec.beginValid && denseSpec.endValid) {
      intervalLength = denseSpec.end[i] - denseSpec.begin[i];
    } else if (shrinkI) {
      // The dimension is still known as 1 for the processingShape, but will be
      // discarded for the final shape.
      intervalLength = 1;
    } else if (beginAndEndMasked) {
      // Even if we don't have values for begin or end, we do know that this
      // dimension covers the whole interval. If we have shape information for
      // this dimension, that tells us the interval length.
      if (dimI >= 0) {
        if (denseSpec.strides[i] < 0) {
          intervalLength = -dimI;
        } else {
          intervalLength = dimI;
        }
      }
    }
    if (intervalLength != null) {
      final int sizeI;
      // Hold zero if the interval is degenerate, otherwise account for
      // remainder
      if (intervalLength == 0 ||
          ((intervalLength < 0) != (denseSpec.strides[i] < 0))) {
        sizeI = 0;
      } else {
        sizeI = (intervalLength / denseSpec.strides[i]).truncate() +
            (intervalLength % denseSpec.strides[i] != 0 ? 1 : 0);
      }
      processingShape.add(sizeI);
    } else {
      processingShape.add(-1);
    }
  }

  // Step 4: Compute the final shape
  //
  // newAxis will increase dimension by 1 (with a one-size dimension)
  // slices like foo[3, ...] will reduce dimension by 1.
  // This cannot be done earlier, because it depends on Step 3.
  for (int denseDim = 0;
      denseDim < denseSpec.finalShapeGatherIndices.length;
      ++denseDim) {
    final gatherIndex = denseSpec.finalShapeGatherIndices[denseDim];
    if (gatherIndex >= 0) {
      finalShape.add(processingShape[gatherIndex]);
    } else if (gatherIndex == NEW_AXIS) {
      finalShape.add(1);
    }
  }

  final finalShapeSparse = finalShape
      .whereIndexed(
          (i, dim) => denseSpec.finalShapeGatherIndices[i] != NEW_AXIS)
      .toList();

  return SliceInfo(
    finalShapeSparse: finalShapeSparse,
    finalShape: finalShape,
    isIdentity: isIdentity,
    sliceDim0: sliceDim0,
    isSimpleSlice: isSimpleSlice,
    begin: denseSpec.begin,
    end: denseSpec.end,
    strides: denseSpec.strides,
  );
}

void _buildDenseSpec(
    _StridedSliceSparseSpec sparse, _StridedSliceDenseSpec dense) {
  // dense.beginMask = 0;
  // dense.endMask = 0;
  // dense.shrinkAxisMask = 0;

  int fullIndex = 0;
  dense.beginValid = sparse.begin != null;
  dense.endValid = sparse.end != null;

  // dense.begin = new Array(dense.dims);
  // dense.end = new Array(dense.dims);
  // dense.strides = new Array(dense.dims);
  // dense.finalShapeGatherIndices = [];
  // dense.finalShapeGatherIndicesSparse = [];
  // dense.inputShapeGatherIndicesSparse = new Array(dense.dims);

  for (int i = 0; i < sparse.dims; i++) {
    if (_intBool((1 << i) & sparse.ellipsisMask)) {
      // Only the bit that has ellipsis will fall in this condition.
      // Expand the ellipsis into the appropriate indices
      // Note: this only works because we guaranteed one ellipsis.
      final nextIndex = math.min(
          dense.dims - (sparse.dims - i) + 1 + sparse.numAddAxisAfterEllipsis,
          dense.dims);
      for (; fullIndex < nextIndex; fullIndex++) {
        // newAxis aren't real axis so you have to skip.
        dense.begin[fullIndex] = 0;
        dense.end[fullIndex] = 0;
        dense.strides[fullIndex] = 1;
        dense.beginMask |= (1 << fullIndex);
        dense.endMask |= (1 << fullIndex);
        dense.finalShapeGatherIndices.add(fullIndex);
        dense.finalShapeGatherIndicesSparse.add(-1);
        dense.inputShapeGatherIndicesSparse[fullIndex] = i;
      }
    } else if (_intBool((1 << i) & sparse.newAxisMask)) {
      // Only the bit that has newAxis will fall in this condition.
      dense.finalShapeGatherIndices.add(NEW_AXIS);
      dense.finalShapeGatherIndicesSparse.add(-1);
    } else {
      if (fullIndex == dense.begin.length) {
        throw Exception(
            "Index out of range using input dim ${fullIndex}; input " +
                "has only ${dense.dims} dims, ${dense.begin.length}.");
      }

      // Gather slicing spec into appropriate index.
      if (sparse.begin != null) {
        dense.begin[fullIndex] = sparse.begin[i];
      }
      if (sparse.end != null) {
        dense.end[fullIndex] = sparse.end[i];
      }
      dense.strides[fullIndex] = sparse.strides[i];
      if (_intBool(sparse.beginMask & (1 << i))) {
        dense.beginMask |= (1 << fullIndex);
      }
      if (_intBool(sparse.endMask & (1 << i))) {
        dense.endMask |= (1 << fullIndex);
      }
      // If shrink, record where to get the dimensionality from (i.e. newAxis)
      // creates a fake 1 size dimension. Also remember shrink axis (now in
      // dense form) so we can ignore dense.end below.
      if (_intBool(sparse.shrinkAxisMask & (1 << i))) {
        dense.finalShapeGatherIndices.add(SHRINK_AXIS);
        dense.finalShapeGatherIndicesSparse.add(-1);
        dense.shrinkAxisMask |= (1 << fullIndex);
      } else {
        dense.finalShapeGatherIndices.add(fullIndex);
        // Remember that where in the sparse shape the dense dim comes from.
        dense.finalShapeGatherIndicesSparse.add(i);
      }
      dense.inputShapeGatherIndicesSparse[fullIndex] = i;
      fullIndex++;
    }
  }
}

int _canonical(int x, int c, int strideI, int dimI, List<int> masks,
    List<int> validRange) {
  if (masks[c] != 0) {
    return strideI > 0 ? validRange[c] : validRange[(c + 1) & 1];
  } else {
    final xFwd = x < 0 ? dimI + x : x; // make negative indices positive
    return xFwd < validRange[0]
        ? validRange[0]
        : xFwd > validRange[1]
            ? validRange[1]
            : xFwd;
  }
}
