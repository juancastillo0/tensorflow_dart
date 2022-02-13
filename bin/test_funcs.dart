import 'dart:io';

import 'package:wasm/wasm.dart';

void main() {
  final moduleBytes = File('./tfjs-backend-wasm.wasm').readAsBytesSync();
  final module = WasmModule(moduleBytes);
  print(module.describe());
  final instance = module.builder().build();
  print(instance.lookupFunction('init'));
  print(instance.lookupFunction('getThreadsCount'));
  print(instance.lookupFunction('cwrap'));
}

// init(): void,
// initWithThreadsCount(threadsCount: number): void,
// getThreadsCount(): number,
// registerTensor(id: number, size: number, memoryOffset: number): void,
// // Disposes the data behind the data bucket.
// disposeData(id: number): void,
// // Disposes the backend and all of its associated data.
// dispose(): void,