import 'dart:io';

import 'package:tensorflow_wasm/src/emscripten_module.dart';
import 'package:wasm/wasm.dart';

void main() async {
  final moduleBytes = File('./tfjs-backend-wasm.wasm').readAsBytesSync();
  final module = WasmModule(moduleBytes);
  print(module.describe());
  // final instance = module.builder().build();
  // print(instance.lookupFunction('init'));
  // print(instance.lookupFunction('getThreadsCount'));
  // print(instance.lookupFunction('cwrap'));

  final mod = await wasmFactory(
    WasmFactoryConfig(wasmBinary: moduleBytes.buffer),
  );
  await Future.delayed(Duration(seconds: 2));
  for (final d in mod.map.entries.where(
      (element) => element.value is! List && element.value is! WasmMemory)) {
    print(d);
  }
}

// init(): void,
// initWithThreadsCount(threadsCount: number): void,
// getThreadsCount(): number,
// registerTensor(id: number, size: number, memoryOffset: number): void,
// // Disposes the data behind the data bucket.
// disposeData(id: number): void,
// // Disposes the backend and all of its associated data.
// dispose(): void,