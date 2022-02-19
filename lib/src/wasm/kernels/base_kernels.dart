import 'package:tensorflow_wasm/src/wasm/kernels/binary_kernel.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/unary_kernel.dart';

import 'package:tensorflow_wasm/src/kernel_names.dart';

// Binary

final addConfig = createBinaryKernelConfig(
  Add,
  supportsFullBroadcast: true,
);
final realDivConfig = createBinaryKernelConfig(
  RealDiv,
  supportsFullBroadcast: true,
);
final squaredDifferenceConfig = createBinaryKernelConfig(
  SquaredDifference,
  supportsFullBroadcast: true,
);
final greaterConfig = createBinaryKernelConfig(
  Greater,
  supportsFullBroadcast: false,
  dtype: 'bool',
);
final lessConfig = createBinaryKernelConfig(
  Less,
  supportsFullBroadcast: false,
  dtype: 'bool',
);
final maximumConfig = createBinaryKernelConfig(
  Maximum,
  supportsFullBroadcast: false,
);
final minimumConfig = createBinaryKernelConfig(
  Minimum,
  supportsFullBroadcast: false,
);
final floorDivConfig = createBinaryKernelConfig(
  FloorDiv,
  supportsFullBroadcast: false,
);
final multiplyConfig = createBinaryKernelConfig(
  Multiply,
  supportsFullBroadcast: true,
);
final powConfig = createBinaryKernelConfig(
  Pow,
  supportsFullBroadcast: false,
);
final subConfig = createBinaryKernelConfig(
  Sub,
  supportsFullBroadcast: true,
);
final equalConfig = createBinaryKernelConfig(
  Equal,
  supportsFullBroadcast: false,
  dtype: 'bool',
);
final greaterEqualConfig = createBinaryKernelConfig(
  GreaterEqual,
  supportsFullBroadcast: false,
  dtype: 'bool',
);
final lessEqualConfig = createBinaryKernelConfig(
  LessEqual,
  supportsFullBroadcast: false,
  dtype: 'bool',
);
final logicalAndConfig = createBinaryKernelConfig(
  LogicalAnd,
  supportsFullBroadcast: false,
  dtype: 'bool',
);
final notEqualConfig = createBinaryKernelConfig(
  NotEqual,
  supportsFullBroadcast: false,
  dtype: 'bool',
);

// Unary

final eluConfig = createUnaryKernelConfig(Elu);
final relu6Config = createUnaryKernelConfig(Relu6);
final reluConfig = createUnaryKernelConfig(Relu);

final coshConfig = createUnaryKernelConfig(Cosh);
final tanConfig = createUnaryKernelConfig(Tan);
final tanhConfig = createUnaryKernelConfig(Tanh);
final cosConfig = createUnaryKernelConfig(Cos);
final sinConfig = createUnaryKernelConfig(Sin);

final negConfig = createUnaryKernelConfig(Neg);
final absConfig = createUnaryKernelConfig(Abs);
final roundConfig = createUnaryKernelConfig(Round);
final floorConfig = createUnaryKernelConfig(Floor);
final ceilConfig = createUnaryKernelConfig(Ceil);

final sqrtConfig = createUnaryKernelConfig(Sqrt);
final rsqrtConfig = createUnaryKernelConfig(Rsqrt);
final squareConfig = createUnaryKernelConfig(Square);
final expConfig = createUnaryKernelConfig(Exp, 'float32');
final logConfig = createUnaryKernelConfig(Log);
