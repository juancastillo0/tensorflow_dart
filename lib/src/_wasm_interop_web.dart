import 'dart:typed_data';

import 'package:wasm/wasm.dart' hide WasmModule;
import 'package:wasm/wasm.dart' as wasm;
import 'package:wasm_interop/wasm_interop.dart';

class WasmModule implements wasm.WasmModule {
  final Module module;
  WasmModule(Uint8List bytes) : module = Module.fromBytes(bytes);

  @override
  WasmInstanceBuilder builder() {
    return _Builder(module);
  }

  @override
  WasmMemory createMemory(int pages, [int? maxPages]) {
    return _Memory(Memory(initial: pages, maximum: maxPages));
  }

  @override
  String describe() {
    return module.imports
        .map((e) => 'import ${e.kind.name}: ${e.module}::${e.name}')
        .followedBy(
          module.exports.map((e) => 'export ${e.kind.name}: ${e.name}'),
        )
        .join('\n');
  }
}

class _Builder implements WasmInstanceBuilder {
  final Module module;
  final Map<String, Map<String, Object>> importMap = {};

  _Builder(this.module);

  @override
  void addFunction(String moduleName, String name, Function fn) {
    importMap.putIfAbsent(moduleName, () => {})[name] = fn;
  }

  @override
  WasmGlobal addGlobal(String moduleName, String name, val) {
    late final Global global;
    if (val is int) {
      global = Global.i32(value: val, mutable: true);
    } else if (val is BigInt) {
      global = Global.i64(value: val, mutable: true);
    } else if (val is double) {
      global = Global.f32(value: val, mutable: true);
    } else {
      global = Global.externref(value: val, mutable: true);
    }
    importMap.putIfAbsent(moduleName, () => {})[name] = global;

    return _Global(global);
  }

  @override
  void addMemory(String moduleName, String name, WasmMemory memory) {
    final Memory _memory = (memory as _Memory).memory;
    importMap.putIfAbsent(moduleName, () => {})[name] = _memory;
  }

  @override
  WasmInstance build() {
    return _Instance(Instance.fromModule(module, importMap: importMap));
  }

  @override
  Future<WasmInstance> buildAsync() async {
    final instance =
        await Instance.fromModuleAsync(module, importMap: importMap);
    return _Instance(instance);
  }

  @override
  void enableWasi({bool captureStdout = false, bool captureStderr = false}) {
    // TODO: implement enableWasi
  }
}

class _Instance implements WasmInstance {
  final Instance instance;

  _Instance(this.instance);

  @override
  Function? lookupFunction(String name) {
    return instance.functions[name];
  }

  @override
  WasmGlobal? lookupGlobal(String name) {
    final global = instance.globals[name];
    if (global == null) return null;
    return _Global(global);
  }

  Map<String, Object> exports(WasmModule module) => {
        ...instance.functions,
        ...instance.globals.map((key, value) => MapEntry(key, _Global(value))),
        ...instance.memories.map((key, value) => MapEntry(key, _Memory(value))),
        ...instance.tables,
      };

  @override
  // TODO: implement memory
  WasmMemory get memory => _Memory(instance.memories.values.first);

  @override
  // TODO: implement stderr
  Stream<List<int>> get stderr => throw UnimplementedError();

  @override
  // TODO: implement stdout
  Stream<List<int>> get stdout => throw UnimplementedError();
}

class _Memory implements WasmMemory {
  final Memory memory;

  _Memory(this.memory);

  @override
  int operator [](int index) {
    return view[index];
  }

  @override
  void operator []=(int index, int value) {
    view[index] = value;
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
  Uint8List get view => Uint8List.view(memory.buffer);
}

class _Global implements WasmGlobal {
  final Global global;

  _Global(this.global);

  @override
  dynamic get value => global.value;

  @override
  set value(dynamic val) {
    global.value = val;
  }
}
