import 'dart:typed_data';

abstract class WasmModule {
  factory WasmModule(Uint8List bytes) {
    throw UnimplementedError();
  }

  WasmInstanceBuilder builder();

  WasmMemory createMemory(int pages, [int? maxPages]);

  String describe();
}

abstract class WasmInstanceBuilder {
  void addFunction(String moduleName, String name, Function fn);

  WasmGlobal addGlobal(String moduleName, String name, val);

  void addMemory(String moduleName, String name, WasmMemory memory);

  WasmInstance build();

  Future<WasmInstance> buildAsync();

  void enableWasi({bool captureStdout = false, bool captureStderr = false});
}

abstract class WasmInstance {
  WasmModule get module;

  Function(List)? lookupFunction(String name);

  WasmGlobal? lookupGlobal(String name);

  Map<String, Object?> exports();

  WasmMemory get memory;

  Stream<List<int>> get stderr;

  Stream<List<int>> get stdout;
}

abstract class WasmMemory {
  int operator [](int index);

  void operator []=(int index, int value);

  void grow(int deltaPages);

  int get lengthInBytes;

  int get lengthInPages;

  Uint8List get view;
}

abstract class WasmGlobal {
  dynamic get value;

  set value(dynamic val);
}
