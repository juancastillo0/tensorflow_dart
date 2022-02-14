export 'package:wasm/wasm.dart' if (dart.library.html) '_wasm_interop_web.dart';

import 'dart:collection';

import 'package:tensorflow_wasm/src/emscripten_module.dart';
import 'package:wasm/wasm.dart';

extension WasmInstanceBuildAsync on WasmInstanceBuilder {
  Future<WasmInstance> buildAsync() async {
    return build();
  }
}

extension WasmInstanceExports on WasmInstance {
  Map<String, Object?> exports(WasmModule module) => _Export(this, module);
}

class WasmInstanceModule {
  final WasmInstance instance;
  final WasmModule module;

  WasmInstanceModule(this.instance, this.module);
}

class _Export extends UnmodifiableMapBase<String, Object?> {
  final WasmInstance instance;
  final WasmModule module;

  _Export(this.instance, this.module);

  @override
  operator [](Object? key) {
    if (key is! String) return null;
    final fn = instance.lookupFunction(key);
    if (fn is WasmFunction) {
      return varArgsFunction((args, _) => fn.apply(args));
    }
    return instance.lookupGlobal(key);
  }

  @override
  Iterable<String> get keys => module
      .describe()
      .split('\n')
      .where((element) => element.startsWith('export'))
      .map((e) => e.contains('(')
          ? e.substring(0, e.indexOf('(')).split(' ').last
          : e.split(' ').last);
}
