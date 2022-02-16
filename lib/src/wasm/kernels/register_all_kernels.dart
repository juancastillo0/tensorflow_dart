import 'package:tensorflow_wasm/src/kernel_registry.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/base_kernels.dart';

void registerAllKernels() {
  final configs = <KernelConfig>[
    addConfig,
    realDivConfig,
    squaredDifferenceConfig,
    greaterConfig,
    lessConfig,
    maximumConfig,
    minimumConfig,
    floorDivConfig,
    multiplyConfig,
    powConfig,
    subConfig,
    equalConfig,
    greaterEqualConfig,
    lessEqualConfig,
    logicalAndConfig,
    notEqualConfig,
    expConfig,
    eluConfig,
    coshConfig,
    tanConfig,
    ceilConfig,
    roundConfig,
    floorConfig,
    negConfig,
    rsqrtConfig,
    relu6Config,
    squareConfig,
    tanhConfig,
    logConfig,
    cosConfig,
    sinConfig,
    absConfig,
    sqrtConfig,
    reluConfig,
  ];

  for (final config in configs) {
    registerKernel(config);
  }
}
