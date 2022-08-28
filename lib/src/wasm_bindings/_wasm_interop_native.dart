import 'dart:typed_data';

import 'dart:collection';

import 'wasm_interface.dart' as wasm;
import 'wasm_interface.dart' hide WasmModule;
import 'package:wasm/wasm.dart' as wasm_io;

Future<WasmModule> compileAsyncWasmModule(Uint8List bytes) async {
  return WasmModule(bytes);
}

class _Export extends UnmodifiableMapBase<String, Object?> {
  final WasmInstance instance;
  final WasmModule module;

  List<String>? _keys;

  _Export(this.instance, this.module);

  @override
  operator [](Object? key) {
    if (key is! String) return null;
    if (key == 'memory') {
      return instance.memory;
    }
    // TODO: support tables
    return instance.lookupFunction(key) ?? instance.lookupGlobal(key);
  }

  @override
  List<String> get keys => _keys ??= module
      .describe()
      .split('\n')
      .where((element) => element.startsWith('export'))
      .map((e) => e.contains('(')
          ? e.substring(0, e.indexOf('(')).split(' ').last
          : e.split(' ').last)
      .toList();
}

class WasmModule implements wasm.WasmModule {
  final wasm_io.WasmModule module;
  WasmModule(Uint8List bytes) : module = wasm_io.WasmModule(bytes);

  @override
  WasmInstanceBuilder builder() {
    return _Builder(module.builder(), this);
  }

  @override
  WasmMemory createMemory(int pages, [int? maxPages]) {
    return _Memory(module.createMemory(pages, maxPages));
  }

  @override
  String describe() {
    return module.describe();
  }
}

class _Builder implements WasmInstanceBuilder {
  final wasm_io.WasmInstanceBuilder builder;

  final WasmModule module;

  _Builder(this.builder, this.module);

  @override
  void addFunction(String moduleName, String name, Function fn) {
    builder.addFunction(moduleName, name, fn);
  }

  @override
  WasmGlobal addGlobal(String moduleName, String name, val) {
    final wasm_io.WasmGlobal global = builder.addGlobal(moduleName, name, val);
    return _Global(global);
  }

  @override
  void addMemory(String moduleName, String name, WasmMemory memory) {
    builder.addMemory(moduleName, name, (memory as _Memory).memory);
  }

  @override
  WasmInstance build() {
    return _Instance(builder.build(), module);
  }

  @override
  Future<WasmInstance> buildAsync() async {
    // TODO:
    return build();
  }

  @override
  void enableWasi({bool captureStdout = false, bool captureStderr = false}) {
    builder.enableWasi(
      captureStdout: captureStdout,
      captureStderr: captureStderr,
    );
  }
}

class _Instance implements WasmInstance {
  final wasm_io.WasmInstance instance;

  @override
  final WasmModule module;

  Map<String, Object?>? _exports;

  _Instance(this.instance, this.module);

  @override
  Function(List)? lookupFunction(String name) {
    return (instance.lookupFunction(name) as wasm_io.WasmFunction?)?.apply;
  }

  @override
  WasmGlobal? lookupGlobal(String name) {
    final global = instance.lookupGlobal(name);
    if (global == null) return null;
    return _Global(global);
  }

  @override
  Map<String, Object?> exports() => _exports ??= _Export(this, module);

  @override
  WasmMemory get memory => _Memory(instance.memory);

  @override
  Stream<List<int>> get stderr => instance.stderr;

  @override
  Stream<List<int>> get stdout => instance.stdout;
}

class _Memory implements WasmMemory {
  final wasm_io.WasmMemory memory;

  _Memory(this.memory);

  @override
  int operator [](int index) {
    return memory[index];
  }

  @override
  void operator []=(int index, int value) {
    memory[index] = value;
  }

  @override
  void grow(int deltaPages) {
    memory.grow(deltaPages);
  }

  @override
  int get lengthInBytes => memory.lengthInBytes;

  @override
  int get lengthInPages => memory.lengthInPages;

  @override
  Uint8List get view => memory.view;
}

class _Global implements WasmGlobal {
  final wasm_io.WasmGlobal global;

  _Global(this.global);

  @override
  dynamic get value => global.value;

  @override
  set value(dynamic val) {
    global.value = val;
  }
}
