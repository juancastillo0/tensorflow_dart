export 'package:tensorflow_wasm/src/backend_wasm.dart';
export 'package:tensorflow_wasm/src/kernel_names.dart';
export 'package:tensorflow_wasm/src/tensor.dart';
export 'package:tensorflow_wasm/src/wasm/kernels/unary_kernel.dart';
export 'package:tensorflow_wasm/src/kernel_registry.dart';

import 'package:tensorflow_wasm/src/tensor.dart';

TensorInfo copyTensorInfo(
  TensorInfo info, {
  DataId? dataId,
  List<int>? shape,
  DataType? dtype,
}) {
  return TensorInfo(
    dataId: dataId ?? info.dataId,
    shape: shape ?? info.shape,
    dtype: dtype ?? info.dtype,
  );
}

List<int> parseAxis(Object value) {
  if (value is List<int>) {
    return value;
  } else if (value is num) {
    if (value is int || value.toInt() == value) {
      return [value.toInt()];
    }
    throw Exception(
      'parseAxis(value=$value) unexpected double.',
    );
  } else if (value is List) {
    return value.map((e) {
      if (e is num && (e is int || e.toInt() == e)) {
        return e.toInt();
      }
      throw Exception(
        'parseAxis(value=$value) unexpected item ${e} of type ${value.runtimeType}.',
      );
    }).toList();
  } else {
    throw Exception(
      'parseAxis(value=$value) unexpected type ${value.runtimeType}.',
    );
  }
}
