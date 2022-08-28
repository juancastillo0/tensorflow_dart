import 'dart:collection';
import 'dart:typed_data';

import 'wasm_interface.dart' as wasm;
import 'wasm_interface.dart' hide WasmModule;
import 'package:wasm_interop/wasm_interop.dart';

Future<wasm.WasmModule> compileAsyncWasmModule(Uint8List bytes) async {
  return WasmModule.compileAsync(bytes);
}

class WasmModule implements wasm.WasmModule {
  final Module module;
  WasmModule(Uint8List bytes) : module = Module.fromBytes(bytes);
  WasmModule._(this.module);

  static Future<wasm.WasmModule> compileAsync(Uint8List bytes) async {
    final module = await Module.fromBytesAsync(bytes);
    return WasmModule._(module);
  }

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
  @override
  late final WasmModule module = WasmModule._(instance.module);

  Map<String, Object>? _exports;

  _Instance(this.instance);

  @override
  Function(List)? lookupFunction(String name) {
    final f = instance.functions[name];
    return f != null ? (List a) => Function.apply(f, a) : null;
  }

  @override
  WasmGlobal? lookupGlobal(String name) {
    final global = instance.globals[name];
    if (global == null) return null;
    return _Global(global);
  }

  @override
  Map<String, Object> exports() =>
      _exports ??= UnmodifiableMapView(Map.fromEntries(
        instance.functions.entries
            .map<MapEntry<String, Object>>(
              (e) => MapEntry(e.key, (List a) => Function.apply(e.value, a)),
            )
            .followedBy(instance.globals.entries
                .map((e) => MapEntry(e.key, _Global(e.value))))
            .followedBy(instance.memories.entries
                .map((e) => MapEntry(e.key, _Memory(e.value))))
            .followedBy(instance.tables.entries),
      ));

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
