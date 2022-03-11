export 'wasm_interface.dart'
    if (dart.library.io) '_wasm_interop_native.dart'
    if (dart.library.html) '_wasm_interop_web.dart';
import 'dart:typed_data';

import 'wasm.dart';

export 'wasm_interface.dart' hide WasmModule;

class WasmInstanceModule {
  final WasmInstance instance;
  final WasmModule module;

  WasmInstanceModule(this.instance, this.module);
}

Future<WasmModule> compileAsyncWasmModule(Uint8List bytes) async {
  if (identical(0, 0.0)) return WasmModule.compileAsync(bytes);
  return WasmModule(bytes);
}
