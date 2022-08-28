import 'dart:io';

import 'package:tensorflow_wasm/src/backend_wasm.dart';
import 'package:wasm/wasm.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

void main() async {
  const wasmPath = './web_examples/web/tensorflow_wasm/tfjs-backend-wasm.wasm';
  final moduleBytes = File(wasmPath).readAsBytesSync();
  final module = WasmModule(moduleBytes);
  print(module.describe());
  // final instance = module.builder().build();
  // print(instance.lookupFunction('init'));
  // print(instance.lookupFunction('getThreadsCount'));
  // print(instance.lookupFunction('cwrap'));

  // final mod = await wasmFactory(
  //   WasmFactoryConfig(wasmBinary: moduleBytes.buffer),
  // );

  setWasmPath(wasmPath);
  await tf.setBackend(wasmBackendFactory);

  for (final d in (tf.backend() as BackendWasm).wasm.map.entries.where(
      (element) => element.value is! List && element.value is! WasmMemory)) {
    print(d);
  }
  tf.add(tf.tensor([2.3, 1, -4]), tf.tensor(5)).print();
}

// init(): void,
// initWithThreadsCount(threadsCount: number): void,
// getThreadsCount(): number,
// registerTensor(id: number, size: number, memoryOffset: number): void,
// // Disposes the data behind the data bucket.
// disposeData(id: number): void,
// // Disposes the backend and all of its associated data.
// dispose(): void,

// ArgMin
// LogicalNot
// LogicalOr
// Reciprocal

// ArgMin
// Bincount
// Complex
// ComplexAbs
// Real
// Conv3D
// Dilation2D
// LinSpace
// LogicalNot
// LogicalOr
// Mod
// Multinomial
// Reciprocal
// ResizeBilinearGrad
// ResizeNearestNeighbor
// ResizeNearestNeighborGrad
// Selu
// Sign
// Softplus
// StringNGrams
// StringSplit
// StringToHashBucketFast
// Unique
// UnsortedSegmentSum
// MaxPoolWithArgMax
// Erf