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
