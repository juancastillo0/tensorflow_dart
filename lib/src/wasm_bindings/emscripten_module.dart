// ignore_for_file: non_constant_identifier_names

import 'dart:async';
import 'dart:convert';
import 'dart:math' as Math;
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:http/http.dart' as http;
import 'package:universal_html/html.dart' as html;
import 'package:universal_io/io.dart' as io;
import 'package:tensorflow_wasm/wasm.dart';

var WasmBackendModuleThreadedSimd = null;

class EmscriptenModule {
  final Map map;
  EmscriptenModule._(this.map);

  EmscriptenModule.fromModule(EmscriptenModule module) : map = module.map;

  Uint8List get HEAPU8 => map['HEAPU8'] as Uint8List;

  Object? Function(List) cwrap(
    String ident,
    String? returnType,
    List<String>? argTypes,
  ) =>
      map['cwrap'](ident, returnType, argTypes);

  int malloc(int size) => map['_malloc']([size]);
  void free(int size) => map['_free']([size]);
}

final Future<EmscriptenModule> Function(WasmFactoryConfig?) wasmFactory = (() {
  var _scriptDir =
      html.document.currentScript?.dir ?? io.Platform.script.toString();
  // if (typeof __filename != "undefined") _scriptDir = _scriptDir || __filename;
  return ([WasmFactoryConfig? WasmBackendModule]) {
    final Map<String, Object?> Module = WasmBackendModule?.toMap() ?? {};

    final readyCompleter = Completer<Map>();
    Module["ready"] = readyCompleter.future;
    Map? beforeListeners;
    // TODO:
    // if (typeof process != "undefined" && process.listeners) {
    //   beforeListeners = {
    //     'uncaughtException': process.listeners("uncaughtException"),
    //     'unhandledRejection': process.listeners("unhandledRejection"),
    //   };
    // }
    Map? moduleOverrides = {};
    // var key;
    for (final key in Module.keys) {
      if (Module.containsKey(key)) {
        moduleOverrides[key] = Module[key];
      }
    }
    var arguments_ = [];
    var thisProgram = "./this.program";
    Function quit_ = (status, toThrow) {
      throw toThrow;
    };
    const ENVIRONMENT_IS_WEB = identical(0, 0.0);
    var ENVIRONMENT_IS_WORKER = false;
    const ENVIRONMENT_IS_NODE = !ENVIRONMENT_IS_WEB;
    var ENVIRONMENT_IS_SHELL = false;

    // ENVIRONMENT_IS_WEB = typeof window == "object";
    ENVIRONMENT_IS_WORKER = ENVIRONMENT_IS_WEB && html.Worker.supported && false
        //  &&
        // html.WorkerGlobalScope.instance.importScripts
        //     is Function
        ;
    // typeof importScripts == "function";
    // ENVIRONMENT_IS_NODE =
    //   typeof process == "object" &&
    //   typeof process.versions == "object" &&
    //   typeof process.versions.node == "string";
    ENVIRONMENT_IS_SHELL =
        !ENVIRONMENT_IS_WEB && !ENVIRONMENT_IS_NODE && !ENVIRONMENT_IS_WORKER;
    var scriptDirectory = "";
    locateFile(String path) {
      if (Module["locateFile"] is Function) {
        return (Module["locateFile"] as Function)(path, scriptDirectory);
      }
      return scriptDirectory + path;
    }

    late Function(String url) readBinary, setWindowTitle;
    late Function(String url, bool binary) read_;
    Function(String url, Function(dynamic) onload,
        Function(Object? err) onerror)? readAsync;
    // var nodeFS;
    // var nodePath;
    if (ENVIRONMENT_IS_NODE) {
      if (ENVIRONMENT_IS_WORKER) {
        // TODO:
        // scriptDirectory = require("path").dirname(scriptDirectory) + "/";
      } else {
        scriptDirectory = io.Platform.script.pathSegments
                .take(io.Platform.script.pathSegments.length - 1)
                .join(io.Platform.pathSeparator) +
            io.Platform.pathSeparator;
      }
      read_ = (String filename, bool binary) {
        // shell_read
        // if (!nodeFS) nodeFS = require("fs");
        // if (!nodePath) nodePath = require("path");
        // filename = nodePath["normalize"](filename);
        // return nodeFS["readFileSync"](filename, binary ? null : "utf8");
        final file = io.File(filename);
        return binary ? file.readAsBytesSync() : file.readAsStringSync();
      };
      readBinary = (String filename) {
        //function readBinary
        var ret = read_(filename, true) as Uint8List;
        // if (!ret.buffer) {
        //   ret = new Uint8Array(ret);
        // }
        // assert(ret.buffer);
        return ret;
      };
      readAsync = (filename, onload, onerror) {
        final file = io.File(filename);
        file
            .readAsBytes()
            .then(onload)
            .onError((error, stackTrace) => onerror(error));
      };
      final argv = io.Platform.executableArguments;
      if (argv.length > 1) {
        thisProgram = argv[1].replaceAll(RegExp(r'\\'), "/");
      }
      arguments_ = argv.length > 2 ? argv.sublist(2) : [];
      // TODO:
      // process["on"]("uncaughtException", (ex) {
      //   if (!(ex is ExitStatus)) {
      //     throw ex;
      //   }
      // });
      // process["on"]("unhandledRejection", abort);
      // quit_ =  (status) {
      //   process["exit"](status);
      // };
      Module["inspect"] = () {
        return "[Emscripten Module object]";
      };
    } else if (ENVIRONMENT_IS_SHELL) {
      // TODO:
      // if (typeof read != "undefined") {
      //   read_ =  (f) { // shell_read
      //     return read(f);
      //   };
      // }
      // readBinary =  (f) { // readBinary
      //   var data;
      //   if (typeof readbuffer == "function") {
      //     return new Uint8Array(readbuffer(f));
      //   }
      //   data = read(f, "binary");
      //   assert(typeof data == "object");
      //   return data;
      // };
      // if (typeof scriptArgs != "undefined") {
      //   arguments_ = scriptArgs;
      // } else if (typeof arguments != "undefined") {
      //   arguments_ = arguments;
      // }
      // if (typeof quit == "function") {
      //   quit_ = (status) {
      //     quit(status);
      //   };
      // }
      // if (typeof print != "undefined") {
      //   if (typeof console == "undefined") console = {};
      //   console.log = print;
      //   console.warn = console.error =
      //     typeof printErr != "undefined" ? printErr : print;
      // }
    } else if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
      if (ENVIRONMENT_IS_WORKER) {
        scriptDirectory = html.WorkerGlobalScope.instance.location.href!;
      } else if (html.document.currentScript != null) {
        scriptDirectory = html.document.currentScript!.src;
      }
      if (_scriptDir != null) {
        scriptDirectory = _scriptDir;
      }
      if (scriptDirectory.indexOf("blob:") != 0) {
        scriptDirectory =
            scriptDirectory.substring(0, scriptDirectory.lastIndexOf("/") + 1);
      } else {
        scriptDirectory = "";
      }

      read_ = (String url, bool binary) {
        var xhr = html.HttpRequest();
        xhr.open("GET", url, async: false);
        xhr.send(null);
        return xhr.responseText;
      };
      if (ENVIRONMENT_IS_WORKER) {
        readBinary = (String url) {
          var xhr = html.HttpRequest();
          xhr.open("GET", url, async: false);
          xhr.responseType = "arraybuffer";
          xhr.send(null);
          return Uint8List.view(xhr.response);
        };
      }
      readAsync =
          (String url, Function(dynamic) onload, Function(Object?) onerror) {
        var xhr = html.HttpRequest();
        xhr.open("GET", url, async: true);
        xhr.responseType = "arraybuffer";
        xhr.onLoad.listen((_) {
          if (xhr.status == 200 || (xhr.status == 0 && xhr.response != null)) {
            onload(xhr.response);
            return;
          }
          onerror('status ${xhr.status}');
        });
        xhr.onError.listen((event) {
          onerror(event.type);
        });
        xhr.send(null);
      };

      setWindowTitle = (title) {
        html.document.title = title;
      };
    } else {}
    final void Function(Object?) out = Module["print"] as Function(Object?)? ??
        print; // TODO: console.log.bind(console);
    final void Function(Object?) err =
        Module["printErr"] as Function(Object?)? ??
            print; // TODO: console.warn.bind(console);
    for (final key in moduleOverrides.keys) {
      if (moduleOverrides.containsKey(key)) {
        Module[key] = moduleOverrides[key];
      }
    }
    moduleOverrides = null;
    if (Module["arguments"] is List) arguments_ = Module["arguments"] as List;
    if (Module["thisProgram"] is String) {
      thisProgram = Module["thisProgram"] as String;
    }
    if (Module["quit"] is Function) quit_ = Module["quit"] as Function;

    ByteBuffer? wasmBinary;
    if (Module["wasmBinary"] is ByteBuffer) {
      wasmBinary = Module["wasmBinary"] as ByteBuffer;
    }
    WasmMemory? wasmMemory;
    var ABORT = false;
    int EXITSTATUS;

    abort([Object? _what]) {
      final onAbort = Module["onAbort"];
      if (onAbort is Function) {
        onAbort(_what);
      }
      String what = _what.toString();
      err(what);
      ABORT = true;
      EXITSTATUS = 1;
      what = "abort(" + what + "). Build with -s ASSERTIONS=1 for more info.";
      // TODO: final e = WebAssembly.RuntimeError(what);
      final e = Exception(what);
      readyCompleter.completeError(e);
      throw e;
    }

    final noExitRuntime = Module["noExitRuntime"] as bool? ?? true;
    // TODO:
    // if (typeof WebAssembly != "object") {
    //   abort("no native wasm support detected");
    // }

    late ByteBuffer buffer;
    late Int8List HEAP8;
    late Uint8List HEAPU8;
    late Int32List HEAP32;
    TypedData HEAP16, HEAPU16, HEAPU32, HEAPF32, HEAPF64;
    UTF8ToString(int? ptr, int? maxBytesToRead) {
      return ptr != null && ptr != 0
          ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead)
          : "";
    }

    stringToUTF8(String str, int outPtr, int maxBytesToWrite) {
      return stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite);
    }

    writeArrayToMemory(List<int> array, int buffer) {
      HEAP8.setRange(buffer, buffer + array.length, array);
    }

    updateGlobalBufferAndViews(ByteBuffer buf) {
      buffer = buf;
      Module["HEAP8"] = HEAP8 = Int8List.view(buf);
      Module["HEAP16"] = HEAP16 = Int16List.view(buf);
      Module["HEAP32"] = HEAP32 = Int32List.view(buf);
      Module["HEAPU8"] = HEAPU8 = Uint8List.view(buf);
      Module["HEAPU16"] = HEAPU16 = Uint16List.view(buf);
      Module["HEAPU32"] = HEAPU32 = Uint32List.view(buf);
      Module["HEAPF32"] = HEAPF32 = Float32List.view(buf);
      Module["HEAPF64"] = HEAPF64 = Float64List.view(buf);
    }

    final int INITIAL_MEMORY = Module["INITIAL_MEMORY"] as int? ?? 16777216;
    var wasmTable;
    late Map<String, dynamic> exports;
    var __ATPRERUN__ = [];
    var __ATINIT__ = [];
    var __ATPOSTRUN__ = [];
    var runtimeInitialized = false;
    var runtimeKeepaliveCounter = 0;
    bool keepRuntimeAlive() {
      return noExitRuntime || runtimeKeepaliveCounter > 0;
    }

    late Function(List) ___wasm_call_ctors;
    ___wasm_call_ctors = (Module["___wasm_call_ctors"] = (arguments) {
      return (___wasm_call_ctors = Module["___wasm_call_ctors"] =
          exports["__wasm_call_ctors"])(arguments);
    });

    var wasmTableMirror = [];
    Function getWasmTableEntry(funcPtr) {
      var func = wasmTableMirror[funcPtr];
      if (func == null) {
        if (funcPtr >= wasmTableMirror.length) {
          // TOOD: wasmTableMirror should be a nullable list
          wasmTableMirror.length = funcPtr + 1;
        }
        wasmTableMirror[funcPtr] = func = wasmTable.get(funcPtr);
      }
      return func;
    }

    callRuntimeCallbacks(List callbacks) {
      while (callbacks.length > 0) {
        final callback = callbacks.removeAt(0);
        if (callback is Function) {
          if (callback is Function(Map<String, Object?>)) callback(Module);
          if (callback is Function()) callback();
          callback([]);
        } else if (callback is Map) {
          final func = callback['func'];
          if (func is int) {
            if (callback['arg'] == null) {
              getWasmTableEntry(func)();
            } else {
              getWasmTableEntry(func)(callback['arg']);
            }
          } else if (func is Function()) {
            func();
          } else {
            func(callback['arg']);
          }
        }
      }
    }

    addOnPreRun(cb) {
      __ATPRERUN__.insert(0, cb);
    }

    void addOnInit(cb) {
      __ATINIT__.insert(0, cb);
    }

    addOnPostRun(cb) {
      __ATPOSTRUN__.insert(0, cb);
    }

    preRun() {
      if (Module["preRun"] != null) {
        if (Module["preRun"] is Function) Module["preRun"] = [Module["preRun"]];
        while ((Module["preRun"] as List).isNotEmpty) {
          addOnPreRun((Module["preRun"] as List).removeAt(0));
        }
      }
      callRuntimeCallbacks(__ATPRERUN__);
    }

    initRuntime() {
      runtimeInitialized = true;
      callRuntimeCallbacks(__ATINIT__);
    }

    postRun() {
      if (Module["postRun"] != null) {
        if (Module["postRun"] is Function) {
          Module["postRun"] = [Module["postRun"]];
        }
        while ((Module["postRun"] as List).isNotEmpty) {
          addOnPostRun((Module["postRun"] as List).removeAt(0));
        }
      }
      callRuntimeCallbacks(__ATPOSTRUN__);
    }

    var runDependencies = 0;
    var runDependencyWatcher = null;
    Function()? dependenciesFulfilled;
    addRunDependency(id) {
      runDependencies++;
      final monitor = Module["monitorRunDependencies"];
      if (monitor is Function) {
        monitor(runDependencies);
      }
    }

    removeRunDependency(id) {
      runDependencies--;
      final monitor = Module["monitorRunDependencies"];
      if (monitor is Function) {
        monitor(runDependencies);
      }
      if (runDependencies == 0) {
        if (runDependencyWatcher != null) {
          // TODO: clearInterval(runDependencyWatcher);
          runDependencyWatcher = null;
        }
        if (dependenciesFulfilled != null) {
          final callback = dependenciesFulfilled!;
          dependenciesFulfilled = null;
          callback();
        }
      }
    }

    //  hasPrefix(String str, prefix) {
    //   return String.prototype.startsWith
    //     ? str.startsWith(prefix)
    //     : str.indexOf(prefix) == 0;
    // }
    var dataURIPrefix = "data:application/octet-stream;base64,";
    isDataURI(String filename) {
      return filename.startsWith(dataURIPrefix);
    }

    var fileURIPrefix = "file://";
    isFileURI(String filename) {
      return filename.startsWith(fileURIPrefix);
    }

    var wasmBinaryFile = "tfjs-backend-wasm.wasm";
    if (!isDataURI(wasmBinaryFile)) {
      wasmBinaryFile = locateFile(wasmBinaryFile);
    }
    Uint8List getBinary(file) {
      try {
        if (file == wasmBinaryFile && wasmBinary != null) {
          return Uint8List.view(wasmBinary);
        }
        if (readBinary != null) {
          return readBinary(file);
        } else {
          throw "both async and sync fetching of the wasm failed";
        }
      } catch (err) {
        abort(err);
      }
    }

    Future<Uint8List> getBinaryPromise() {
      if (wasmBinary == null && (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER)) {
        if (!isFileURI(wasmBinaryFile)) {
          return http
              .get(
            Uri.parse(wasmBinaryFile),
            // headers: {'credentials': "same-origin"},
          )
              .then((response) {
            if (response.statusCode >= 300) {
              throw ("failed to load wasm binary file at '$wasmBinaryFile'");
            }
            return response.bodyBytes;
          }).catchError((_) {
            return getBinary(wasmBinaryFile);
          });
        } else {
          if (readAsync != null) {
            final completer = Completer<Uint8List>();
            readAsync(
              wasmBinaryFile,
              (response) {
                completer.complete(response is Uint8List
                    ? response
                    : Uint8List.view(response));
              },
              (err) => completer.completeError(err ?? ''),
            );
            return completer.future;
          }
        }
      }
      return Future(() {
        return getBinary(wasmBinaryFile);
      });
    }

    Module["preloadedImages"] = {};
    Module["preloadedAudios"] = {};

    late Function(List) ___errno_location;
    ___errno_location = (Module["___errno_location"] = (arguments) {
      return (___errno_location =
          Module["___errno_location"] = exports["__errno_location"])(arguments);
    });

    final asmLibraryArg = asmLibraryArgs(
      () => wasmMemory!,
      abort,
      (size) {
        try {
          wasmMemory!.grow((size - buffer.lengthInBytes + 65535) >>> 16);
          updateGlobalBufferAndViews(wasmMemory!.view.buffer);
          return 1;
        } catch (_) {
          return null;
        }
      },
      (msg, {required isError}) {
        if (isError) {
          err(msg);
        } else {
          out(msg);
        }
      },
      () => ___errno_location([]),
    );

    createWasm() {
      final info = {
        'env': asmLibraryArg,
        'wasi_snapshot_preview1': asmLibraryArg
      };
      receiveInstance(WasmInstance instance, WasmModule module) {
        exports = instance.exports(module);
        Module["asm"] = exports;
        // TODO: wasmMemory = Module["asm"]["j"];
        wasmMemory = instance.memory;
        updateGlobalBufferAndViews(wasmMemory!.view.buffer);
        // TODO: wasmTable = Module["asm"]["r"];
        addOnInit((Module["asm"] as Map)["__wasm_call_ctors"]);
        removeRunDependency("wasm-instantiate");
      }

      addRunDependency("wasm-instantiate");
      receiveInstantiatedSource(WasmInstanceModule output) {
        receiveInstance(output.instance, output.module);
      }

      instantiateArrayBuffer(Function(WasmInstanceModule) receiver) {
        return getBinaryPromise().then((binary) async {
          // return WebAssembly.instantiate(binary, info);
          // return wasm_interop.Instance.fromBytesAsync(binary, importMap: info);
          final module = await compileAsyncWasmModule(binary);
          print(module.describe());
          final builder = module.builder();
          for (final modEntry in info.entries) {
            for (final entry in modEntry.value.entries) {
              builder.addFunction(modEntry.key, entry.key, entry.value);
            }
          }
          return builder
              .buildAsync()
              .then((value) => WasmInstanceModule(value, module));
        }).then(receiver, onError: (reason) {
          err("failed to asynchronously prepare wasm: " + reason.toString());
          abort(reason);
        });
      }

      Future instantiateAsync() {
        // TODO: WebAssembly.instantiateStreaming
        // if (
        //   wasmBinary != null &&
        //   WebAssembly.instantiateStreaming == "function" &&
        //   !isDataURI(wasmBinaryFile) &&
        //   !isFileURI(wasmBinaryFile) &&
        //   fetch == "function"
        // ) {
        //   return fetch(wasmBinaryFile, { 'credentials': "same-origin" }).then(
        //     (response) {
        //       var result = WebAssembly.instantiateStreaming(response, info);
        //       return result.then(receiveInstantiatedSource, (reason) {
        //         err("wasm streaming compile failed: " + reason);
        //         err("falling back to ArrayBuffer instantiation");
        //         return instantiateArrayBuffer(receiveInstantiatedSource);
        //       });
        //     }
        //   );
        // } else {
        return instantiateArrayBuffer(receiveInstantiatedSource);
        // }
      }

      if (Module["instantiateWasm"] is Function) {
        try {
          var exports =
              (Module["instantiateWasm"] as Function)(info, receiveInstance);
          return exports;
        } catch (e) {
          err("Module.instantiateWasm callback failed with error: " +
              e.toString());
          return false;
        }
      }
      instantiateAsync().catchError(readyCompleter.completeError);
      return {};
    }

    final asm = createWasm();

    (Module["_init"] = ((arguments) {
      return (Module["_init"] = exports["init"])(arguments);
    }));
    (Module["_init_with_threads_count"] = ((arguments) {
      return (Module["_init_with_threads_count"] =
          exports["init_with_threads_count"])(arguments);
    }));
    (Module["_get_threads_count"] = ((arguments) {
      return (Module["_get_threads_count"] =
          exports["get_threads_count"])(arguments);
    }));
    (Module["_register_tensor"] = ((arguments) {
      return (Module["_register_tensor"] =
          exports["register_tensor"])(arguments);
    }));
    (Module["_dispose_data"] = ((arguments) {
      return (Module["_dispose_data"] = exports["dispose_data"])(arguments);
    }));
    (Module["_dispose"] = ((arguments) {
      return (Module["_dispose"] = exports["dispose"])(arguments);
    }));

    addTensorFlowFunctions(Module);

    (Module["_malloc"] = ((arguments) {
      return (Module["_malloc"] = exports["malloc"])(arguments);
    }));
    (Module["_free"] = ((arguments) {
      return (Module["_free"] = exports["free"])(arguments);
    }));
    late Function(List) stackSave;
    stackSave = (Module["stackSave"] = (args) {
      return (stackSave = Module["stackSave"] = exports["stackSave"])(args);
    });
    late Function(List) stackRestore;
    stackRestore = (Module["stackRestore"] = ((arguments) {
      return ((stackRestore =
          Module["stackRestore"] = exports["stackRestore"]))(arguments);
    }));
    late Function(List) stackAlloc;
    stackAlloc = (Module["stackAlloc"] = ((arguments) {
      return ((stackAlloc =
          Module["stackAlloc"] = exports["stackAlloc"]))(arguments);
    }));

    assertC(bool condition, String text) {
      if (!condition) {
        abort("Assertion failed: " + text);
      }
    }

    Function(List) getCFunc(String ident) {
      var func = Module["_" + ident];
      assertC(
        func is Function(List),
        "Cannot call unknown function " + ident + ", make sure it is exported",
      );
      return func as Function(List);
    }

    final toC = {
      'string': (str) {
        var ret = 0;
        if (str is String) {
          // && str != undefined
          var len = (str.length << 2) + 1;
          ret = stackAlloc([len]);
          stringToUTF8(str, ret, len);
        }
        return ret;
      },
      'array': (arr) {
        var ret = stackAlloc([(arr as List).length]);
        writeArrayToMemory(arr as List<int>, ret);
        return ret;
      },
    };
    ccall(
      String ident,
      String? returnType,
      List<String> argTypes,
      List? args,
      Object? opts,
    ) {
      convertReturnValue(Object? ret) {
        if (returnType == "string") return UTF8ToString(ret as int, null);
        if (returnType == "boolean") {
          return ret != null && ret != 0 && ret != '';
        }
        return ret;
      }

      var func = getCFunc(ident);
      var cArgs = [];
      var stack = 0;
      if (args != null) {
        for (var i = 0; i < args.length; i++) {
          var converter = toC[argTypes[i]];
          if (converter != null) {
            if (stack == 0) stack = stackSave([]);
            cArgs.add(converter(args[i]));
          } else {
            cArgs.add(args[i]);
          }
        }
      }
      var ret = func(cArgs);
      Object? onDone(ret) {
        if (stack != 0) stackRestore([stack]);
        return convertReturnValue(ret);
      }

      ret = onDone(ret);
      return ret;
    }

    Function(List) cwrap(
      String ident,
      String? returnType,
      List<String>? argTypes,
      // TODO: Object? opts,
    ) {
      argTypes = argTypes ?? [];
      var numericArgs = argTypes.every((type) {
        return type == "number";
      });
      var numericRet = returnType != "string";
      if (numericRet && numericArgs) {
        // && opts == null
        return getCFunc(ident);
      }
      return (arguments) {
        return ccall(ident, returnType, argTypes!, arguments, null);
      };
    }

    Module["cwrap"] = cwrap;
    bool calledRun = false;

    run([args]) {
      args = args ?? arguments_;
      if (runDependencies > 0) {
        return;
      }
      preRun();
      if (runDependencies > 0) {
        return;
      }
      doRun() {
        if (calledRun) return;
        calledRun = true;
        Module["calledRun"] = true;
        if (ABORT) return;
        initRuntime();
        readyCompleter.complete(Module);
        if (Module["onRuntimeInitialized"] is Function) {
          (Module["onRuntimeInitialized"] as Function)();
        }
        postRun();
      }

      final setStatus = Module["setStatus"];
      if (setStatus is Function(String)) {
        setStatus("Running...");
        Timer(const Duration(milliseconds: 1), () {
          Timer(const Duration(milliseconds: 1), () {
            setStatus("");
          });
          doRun();
        });
      } else {
        doRun();
      }
    }

    runCaller() {
      if (!calledRun) run();
      if (!calledRun) dependenciesFulfilled = runCaller;
    }

    dependenciesFulfilled = runCaller;
    Module["run"] = run;
    if (Module["preInit"] != null) {
      if (Module["preInit"] is Function) {
        Module["preInit"] = [Module["preInit"]];
      }
      final preInit = Module["preInit"] as List;
      while (preInit.isNotEmpty) {
        preInit.removeLast()();
      }
    }
    run();
    // TODO:
    // Map? listenersAdded;
    // if (beforeListeners != null) {
    //   listenersAdded = {
    //     'uncaughtException':
    //         process.listeners("uncaughtException").filter((listener) {
    //       return beforeListeners.uncaughtException.indexOf(listener) <= -1;
    //     }),
    //     'unhandledRejection':
    //         process.listeners("unhandledRejection").filter((listener) {
    //       return beforeListeners.unhandledRejection.indexOf(listener) <= -1;
    //     }),
    //   };
    // }
    var actualModule;
    if (WasmBackendModule != null) {
      actualModule = WasmBackendModule;
    } else if (WasmBackendModuleThreadedSimd != null) {
      actualModule = WasmBackendModuleThreadedSimd;
    } else {
      throw Exception("Could not find wasm module in post.js");
    }
    // TODO:
    // if (listenersAdded != null) {
    //   var tmpDispose = actualModule["_dispose"];
    //   actualModule["_dispose"] = () {
    //     tmpDispose();
    //     listenersAdded.uncaughtException.forEach((listener) {
    //       process.removeListener("uncaughtException", listener);
    //     });
    //     listenersAdded.unhandledRejection.forEach((listener) {
    //       process.removeListener("unhandledRejection", listener);
    //     });
    //   };
    // }

    return readyCompleter.future.then((value) => EmscriptenModule._(value));
  };
})();

// if (typeof exports == "object" && typeof module == "object")
//   module.exports = WasmBackendModule;
// else if (typeof define == "function" && define["amd"])
//   define([], function () {
//     return WasmBackendModule;
//   });
// else if (typeof exports == "object")
//   exports["WasmBackendModule"] = WasmBackendModule;

String UTF8ArrayToString(Uint8List heap, int idx, int? maxBytesToRead) {
  var endIdx = maxBytesToRead != null ? idx + maxBytesToRead : heap.length;
  var endPtr = idx;
  while (endPtr < endIdx && heap[endPtr] != 0) {
    ++endPtr;
  }
  if (endPtr - idx > 16) {
    return utf8.decode(heap.slice(idx, endPtr)); // TODO: view
  } else {
    var str = "";
    while (idx < endPtr) {
      var u0 = heap[idx++];
      if (u0 & 128 == 0) {
        str += String.fromCharCode(u0);
        continue;
      }
      var u1 = heap[idx++] & 63;
      if ((u0 & 224) == 192) {
        str += String.fromCharCode(((u0 & 31) << 6) | u1);
        continue;
      }
      var u2 = heap[idx++] & 63;
      if ((u0 & 240) == 224) {
        u0 = ((u0 & 15) << 12) | (u1 << 6) | u2;
      } else {
        u0 = ((u0 & 7) << 18) | (u1 << 12) | (u2 << 6) | (heap[idx++] & 63);
      }
      if (u0 < 65536) {
        str += String.fromCharCode(u0);
      } else {
        var ch = u0 - 65536;
        str += String.fromCharCodes([55296 | (ch >> 10), 56320 | (ch & 1023)]);
      }
    }
    return str;
  }
}

stringToUTF8Array(String str, Uint8List heap, int outIdx, int maxBytesToWrite) {
  if (!(maxBytesToWrite > 0)) return 0;
  var startIdx = outIdx;
  var endIdx = outIdx + maxBytesToWrite - 1;
  for (var i = 0; i < str.length; ++i) {
    var u = str.codeUnitAt(i); // TODO: was charCodeAt
    if (u >= 55296 && u <= 57343) {
      var u1 = str.codeUnitAt(++i);
      u = (65536 + ((u & 1023) << 10)) | (u1 & 1023);
    }
    if (u <= 127) {
      if (outIdx >= endIdx) break;
      heap[outIdx++] = u;
    } else if (u <= 2047) {
      if (outIdx + 1 >= endIdx) break;
      heap[outIdx++] = 192 | (u >> 6);
      heap[outIdx++] = 128 | (u & 63);
    } else if (u <= 65535) {
      if (outIdx + 2 >= endIdx) break;
      heap[outIdx++] = 224 | (u >> 12);
      heap[outIdx++] = 128 | ((u >> 6) & 63);
      heap[outIdx++] = 128 | (u & 63);
    } else {
      if (outIdx + 3 >= endIdx) break;
      heap[outIdx++] = 240 | (u >> 18);
      heap[outIdx++] = 128 | ((u >> 12) & 63);
      heap[outIdx++] = 128 | ((u >> 6) & 63);
      heap[outIdx++] = 128 | (u & 63);
    }
  }
  heap[outIdx] = 0;
  return outIdx - startIdx;
}

alignUp(int x, int multiple) {
  if (x % multiple > 0) {
    x += multiple - (x % multiple);
  }
  return x;
}

Map<String, Function> asmLibraryArgs(
  WasmMemory Function() wasmMemory,
  Function abort,
  int? Function(int size) reallocBuffer,
  void Function(String message, {required bool isError}) printMsg,
  int Function() ___errno_location,
) {
  _abort() {
    abort('');
  }

  int _emscripten_get_heap_max() {
    return 2147483648;
  }

  _emscripten_memcpy_big(int dest, int src, int num) {
    // TODO: HEAPU8.copyWithin(dest, src, src + num);
    final HEAPU8 = wasmMemory().view;
    // HEAPU8.setAll(dest, HEAPU8.sublist(src, src + num));
    List.copyRange(HEAPU8, dest, HEAPU8, src, src + num);
  }

  emscripten_realloc_buffer(int size) {
    return reallocBuffer(size);
  }

  _emscripten_resize_heap(int requestedSize) {
    final oldSize = wasmMemory().view.length;
    requestedSize = requestedSize >>> 0;
    final maxHeapSize = _emscripten_get_heap_max();
    if (requestedSize > maxHeapSize) {
      return false;
    }
    for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
      int overGrownHeapSize = (oldSize * (1 + 0.2 / cutDown)).toInt();
      overGrownHeapSize =
          Math.min(overGrownHeapSize, requestedSize + 100663296);
      final int newSize = Math.min(maxHeapSize,
          alignUp(Math.max(requestedSize, overGrownHeapSize), 65536));
      var replacement = emscripten_realloc_buffer(newSize);
      if (replacement != null && replacement != 0) {
        return true;
      }
    }
    return false;
  }

  final List<List<int>?> SYSCALLSbuffers = [null, [], []];
  SYSCALLSprintChar(int stream, int curr) {
    final buffer = SYSCALLSbuffers[stream]!;
    if (curr == 0 || curr == 10) {
      printMsg(
        UTF8ArrayToString(Uint8List.fromList(buffer), 0, null),
        isError: stream != 1,
      );
      buffer.length = 0;
    } else {
      buffer.add(curr);
    }
  }

  // var SYSCALLS = {
  //   'mappings': {},
  //   'buffers': [null, [], []],
  //   'printChar': (stream, curr) {
  //     var buffer = SYSCALLSbuffers[stream];
  //     if (curr == 0 || curr == 10) {
  //       (stream == 1 ? out : err)(UTF8ArrayToString(buffer, 0));
  //       buffer.length = 0;
  //     } else {
  //       buffer.push(curr);
  //     }
  //   },
  //   // 'varargs': null, // TODO: was undefined
  //   // 'get': () {
  //   //   SYSCALLS.varargs += 4;
  //   //   var ret = HEAP32[(SYSCALLS.varargs - 4) >> 2];
  //   //   return ret;
  //   // },
  //   // 'getStr': (ptr) {
  //   //   var ret = UTF8ToString(ptr);
  //   //   return ret;
  //   // },
  //   // 'get64': (low, high) {
  //   //   return low;
  //   // },
  // };
  _fd_close(fd) {
    return 0;
  }

  _fd_seek(fd, offset_low, offset_high, whence, newOffset) {}
  _fd_write(fd, iov, iovcnt, pnum) {
    var num = 0;
    final HEAP32 = Int32List.view(wasmMemory().view.buffer);
    for (var i = 0; i < iovcnt; i++) {
      final ptr = HEAP32[iov >> 2];
      final len = HEAP32[(iov + 4) >> 2];
      iov += 8;
      for (var j = 0; j < len; j++) {
        SYSCALLSprintChar(fd, wasmMemory().view[ptr + j]);
      }
      num += len;
    }
    HEAP32[pnum >> 2] = num;
    return 0;
  }

  var tempRet0 = 0;
  var setTempRet0 = (value) {
    tempRet0 = value;
  };

  void _setTempRet0(val) {
    setTempRet0(val);
  }

  return {
    'abort': _abort,
    'emscripten_get_heap_max': _emscripten_get_heap_max,
    'emscripten_memcpy_big': _emscripten_memcpy_big,
    'emscripten_resize_heap': _emscripten_resize_heap,
    'fd_close': _fd_close,
    'fd_seek': _fd_seek,
    'fd_write': _fd_write,
    'setTempRet0': _setTempRet0,
  };
}

void addTensorFlowFunctions(Map Module) async {
  for (final entry in tfNames.entries) {
    Module[entry.key] = (args) {
      try {
        return ((Module[entry.key] = Module['asm'][entry.key.substring(1)])
            as Function)(args);
      } catch (e, s) {
        print('addTensorFlowFunctions ${entry.key} ${args} $e $s');
        rethrow;
      }
    };
  }
}

const tfNames = {
  '_Abs': 's', // done
  '_Add': 't', // done
  '_AddN': 'u',
  '_All': 'v', // done
  '_Any': 'w', // done
  '_ArgMax': 'x', // new
  '_AvgPool': 'y',
  '_BatchMatMul': 'z', // new
  '_Ceil': 'A', // done
  '_ClipByValue': 'B',
  '_Conv2D': 'C',
  '_Conv2DBackpropInput': 'D',
  '_Cos': 'E', // done
  '_Cosh': 'F', // done
  '_CropAndResize': 'G',
  '_Cumsum': 'H', // new
  '_DepthToSpace': 'I',
  '_DepthwiseConv2dNative': 'J',
  '_Elu': 'K', // done
  '_Equal': 'L', // done
  '_Exp': 'M', // done
  '_FlipLeftRight': 'N',
  '_Floor': 'O', // done
  '_FloorDiv': 'P', // done
  '_FusedBatchNorm': 'Q', // new
  '_FusedConv2D': 'R', // new
  '_FusedDepthwiseConv2D': 'S', // new
  '_Gather': 'T',
  '_GatherNd': 'U',
  '_Greater': 'V', // done
  '_GreaterEqual': 'W', // done
  '_LeakyRelu': 'X',
  '_Less': 'Y', // done
  '_LessEqual': 'Z', // done
  '_Log': '_', // done
  '_LogicalAnd': r'$', // done
  '_Max': 'aa', // new
  '_MaxPool': 'ba',
  '_Maximum': 'ca', // new
  '_Mean': 'da', // new
  '_Min': 'ea', // new
  '_Minimum': 'fa', // new
  '_MirrorPad': 'ga',
  '_Multiply': 'ha', // done
  '_Neg': 'ia', // done
  '_NonMaxSuppressionV3': 'ja',
  '_NonMaxSuppressionV4': 'ka',
  '_NonMaxSuppressionV5': 'la',
  '_NotEqual': 'ma', // done
  '_OneHot': 'na', // new
  '_PadV2': 'oa',
  '_Pow': 'pa', // done
  '_Prelu': 'qa',
  '_Prod': 'ra', // new
  '_RealDiv': 'sa', // done
  '_Relu': 'ta', // done
  '_Relu6': 'ua', // done
  '_ResizeBilinear': 'va',
  '_Reverse': 'wa',
  '_RotateWithOffset': 'xa',
  '_Round': 'ya', // done
  '_Rsqrt': 'za', // done
  '_ScatterNd': 'Aa',
  '_SelectV2': 'Ba', // done
  '_Sigmoid': 'Ca', // new
  '_Sin': 'Da', // done
  '_Softmax': 'Ea', // new
  '_SparseFillEmptyRows': 'Fa',
  '_SparseReshape': 'Ga',
  '_SparseSegmentReduction': 'Ha',
  '_Sqrt': 'Ia', // done
  '_Square': 'Ja', // done
  '_SquaredDifference': 'Ka', // done
  '_Step': 'La',
  '_StridedSlice': 'Ma',
  '_Sub': 'Na', // done
  '_Sum': 'Oa', // new
  '_Tan': 'Pa', // done
  '_Tanh': 'Qa', // done
  '_Tile': 'Ra', // done
  '_TopK': 'Sa', // new
  '_Transform': 'Ta',
  '_Transpose': 'Ua', // done
  '__FusedMatMul': 'Va' // new
};

class WasmFactoryConfig {
  final Object? mainScriptUrlOrBlob; //: string|Blob;
  final String Function(String path, String prefix)? locateFile;
  final Function(
    Map<String, Map<String, Function>> info,
    void Function(WasmInstance, WasmModule),
  )? instantiateWasm;
  final void Function()? onRuntimeInitialized;
  final void Function(Object?)? onAbort;
  final ByteBuffer? wasmBinary;

  const WasmFactoryConfig({
    this.mainScriptUrlOrBlob,
    this.locateFile,
    this.instantiateWasm,
    this.onRuntimeInitialized,
    this.onAbort,
    this.wasmBinary,
  });

  Map<String, Object?> toMap() {
    return {
      'mainScriptUrlOrBlob': mainScriptUrlOrBlob,
      'locateFile': locateFile,
      'instantiateWasm': instantiateWasm,
      'onRuntimeInitialized': onRuntimeInitialized,
      'onAbort': onAbort,
      'wasmBinary': wasmBinary,
    };
  }

  factory WasmFactoryConfig.fromMap(Map<dynamic, dynamic> map) {
    return WasmFactoryConfig(
      mainScriptUrlOrBlob: map['mainScriptUrlOrBlob'],
      locateFile: map['locateFile'],
      instantiateWasm: map['instantiateWasm'],
      onRuntimeInitialized: map['onRuntimeInitialized'],
      onAbort: map['onAbort'],
      wasmBinary: map['wasmBinary'],
    );
  }
}

class ExitStatus implements Exception {
  final name = "ExitStatus";
  String get message => "Program terminated with exit(" + status + ")";
  final String status;

  const ExitStatus(this.status);
}

// typedef VarArgsCallback<T> = T Function(
//     List<dynamic> args, Map<String, Object?> kwargs);

// Function varArgsFunction<T>(VarArgsCallback<T> callback) {
//   return VarArgsFunction._(callback);
// }

// class VarArgsFunction<T> {
//   final VarArgsCallback<T> callback;
//   static final _symbolOffset = 'Symbol("'.length;

//   VarArgsFunction._(this.callback);

//   T call() => callback([], {});

//   @override
//   dynamic noSuchMethod(Invocation inv) {
//     return callback(
//       inv.positionalArguments,
//       inv.namedArguments.map(
//         (_k, v) {
//           var k = _k.toString();
//           return MapEntry(k.substring(_symbolOffset, k.length - 2), v);
//         },
//       ),
//     );
//   }
// }
