import 'dart:typed_data';

import 'package:collection/collection.dart' hide ListExtensions;
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';
import 'package:tensorflow_wasm/slice_util.dart' as slice_util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;
import 'package:tensorflow_wasm/src/util_base.dart' as util;

BackendValues sliceImplCPU(
    BackendValues vals, List<int> begin, List<int> size, List<int> shape,
    DataType dtype,) {
  final isContinous = slice_util.isSliceContinous(shape, begin, size);
  final length = util.sizeFromShape(size);
  final xStrides = util.computeStrides(shape);

  if (isContinous) {
    final flatOffset = slice_util.computeFlatOffset(begin, xStrides);

    if (dtype == 'string') {
      return (vals as List<Uint8List>).slice(flatOffset, flatOffset + length);
    }

    return (vals as TypedArray).subarray(flatOffset, flatOffset + length);
  }

  final decodedData = dtype == 'string' ?
      backend_util.fromUint8ToStringArray(vals as List<Uint8List>) :
      vals as TypedArray;

  final inBuf = buffer(shape, dtype, decodedData);
  final outBuf = buffer(size, dtype);
  for (int i = 0; i < outBuf.size; ++i) {
    final outLoc = outBuf.indexToLoc(i);
    final inLoc = outLoc.mapIndexed((j, idx) => idx + begin[j]);
    outBuf.set(inBuf.get(...inLoc), ...outLoc);
  }

  if (dtype == 'string') {
    return backend_util.fromStringArrayToUint8(outBuf.values as string[]);
  }
  return outBuf.values as TypedArray;
}