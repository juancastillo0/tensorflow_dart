// ignore_for_file: non_constant_identifier_names

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as Math;
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:universal_html/html.dart' as html;
import 'package:universal_io/io.dart' as io;
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

var WasmBackendModuleThreadedSimd = null;

class EmscriptenModule {
  final Map map;
  EmscriptenModule._(this.map);

  EmscriptenModule.fromModule(EmscriptenModule module) : map = module.map;

  Uint8List get HEAPU8 => map['HEAPU8'] as Uint8List;

  Function cwrap(
    String ident,
    String? returnType,
    List<String>? argTypes,
  ) =>
      map['cwrap'](
        ident,
        returnType,
        argTypes,
      );

  int malloc(int size) => map['_malloc'](size);
  void free(int size) => map['_free'](size);
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
    ENVIRONMENT_IS_WORKER = ENVIRONMENT_IS_WEB &&
        html.Worker.supported &&
        html.WorkerGlobalScope.instance.importScripts
            is Function; // typeof importScripts == "function";
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
        scriptDirectory = Platform.script.pathSegments
                .take(Platform.script.pathSegments.length - 1)
                .join(Platform.pathSeparator) +
            Platform.pathSeparator;
      }
      read_ = (String filename, bool binary) {
        // shell_read
        // if (!nodeFS) nodeFS = require("fs");
        // if (!nodePath) nodePath = require("path");
        // filename = nodePath["normalize"](filename);
        // return nodeFS["readFileSync"](filename, binary ? null : "utf8");
        final file = File(filename);
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

    final noExitRuntime = Module["noExitRuntime"] ?? true;
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
    var __ATMAIN__ = [];
    var __ATPOSTRUN__ = [];
    var runtimeInitialized = false;

    late Function ___wasm_call_ctors;
    ___wasm_call_ctors =
        (Module["___wasm_call_ctors"] = varArgsFunction((arguments, _) {
      return Function.apply(
          ___wasm_call_ctors = Module["___wasm_call_ctors"] = exports["k"],
          arguments);
    }));
    __ATINIT__.add({
      'func': () {
        ___wasm_call_ctors();
      },
    });

    callRuntimeCallbacks(List callbacks) {
      while (callbacks.length > 0) {
        final callback = callbacks.removeAt(0);
        if (callback is Function) {
          callback(Module);
        } else if (callback is Map) {
          final func = callback['func'];
          if (func is int) {
            if (callback['arg'] == null) {
              wasmTable.get(func)();
            } else {
              wasmTable.get(func)(callback['arg']);
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

    preMain() {
      callRuntimeCallbacks(__ATMAIN__);
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
          return http.get(Uri.parse(wasmBinaryFile),
              headers: {'credentials': "same-origin"}).then((response) {
            if (response.statusCode >= 300) {
              throw ("failed to load wasm binary file at '" +
                  wasmBinaryFile +
                  "'");
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
                completer.complete(Uint8List(response));
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

    late Function ___errno_location;
    ___errno_location =
        (Module["___errno_location"] = varArgsFunction((arguments, _) {
      return Function.apply(
          ___errno_location = Module["___errno_location"] = exports["Ya"],
          arguments);
    }));

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
      () => ___errno_location(),
    );

    createWasm() {
      var info = {'a': asmLibraryArg};
      receiveInstance(WasmInstance instance, WasmModule module) {
        exports = instance.exports(module);
        Module["asm"] = exports;
        // TODO: wasmMemory = Module["asm"]["j"];
        wasmMemory = instance.memory;
        updateGlobalBufferAndViews(wasmMemory!.view.buffer);
        // TODO: wasmTable = Module["asm"]["r"];
        removeRunDependency("wasm-instantiate");
      }

      addRunDependency("wasm-instantiate");
      receiveInstantiatedSource(WasmInstanceModule output) {
        receiveInstance(output.instance, output.module);
      }

      instantiateArrayBuffer(Function(WasmInstanceModule) receiver) {
        return getBinaryPromise().then((binary) {
          // return WebAssembly.instantiate(binary, info);
          // return wasm_interop.Instance.fromBytesAsync(binary, importMap: info);
          final module = WasmModule(binary);
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

    (Module["_init"] = varArgsFunction((arguments, _) {
      return Function.apply(Module["_init"] = exports["l"], arguments);
    }));
    (Module["_init_with_threads_count"] = varArgsFunction((arguments, _) {
      return Function.apply(
          Module["_init_with_threads_count"] = exports["m"], arguments);
    }));
    (Module["_get_threads_count"] = varArgsFunction((arguments, _) {
      return Function.apply(
          Module["_get_threads_count"] = exports["n"], arguments);
    }));
    (Module["_register_tensor"] = varArgsFunction((arguments, _) {
      return Function.apply(
          Module["_register_tensor"] = exports["o"], arguments);
    }));
    (Module["_dispose_data"] = varArgsFunction((arguments, _) {
      return Function.apply(Module["_dispose_data"] = exports["p"], arguments);
    }));
    (Module["_dispose"] = varArgsFunction((arguments, _) {
      return Function.apply(Module["_dispose"] = exports["q"], arguments);
    }));

    addTensorFlowFunctions(Module);

    (Module["_malloc"] = varArgsFunction((arguments, _) {
      return Function.apply(Module["_malloc"] = exports["Wa"], arguments);
    }));
    (Module["_free"] = varArgsFunction((arguments, _) {
      return Function.apply(Module["_free"] = exports["Xa"], arguments);
    }));
    late Function() stackSave;
    stackSave = (Module["stackSave"] = () {
      return (stackSave = Module["stackSave"] = exports["Za"])();
    });
    late Function stackRestore;
    stackRestore = (Module["stackRestore"] = varArgsFunction((arguments, _) {
      return Function.apply(
          (stackRestore = Module["stackRestore"] = exports["_a"]), arguments);
    }));
    late Function stackAlloc;
    stackAlloc = (Module["stackAlloc"] = varArgsFunction((arguments, _) {
      return Function.apply(
          (stackAlloc = Module["stackAlloc"] = exports[r"$a"]), arguments);
    }));

    assertC(bool condition, String text) {
      if (!condition) {
        abort("Assertion failed: " + text);
      }
    }

    Function getCFunc(String ident) {
      var func = Module["_" + ident];
      assertC(
        func is Function,
        "Cannot call unknown function " + ident + ", make sure it is exported",
      );
      return func as Function;
    }

    final toC = {
      'string': (str) {
        var ret = 0;
        if (str is String) {
          // && str != undefined
          var len = (str.length << 2) + 1;
          ret = stackAlloc(len);
          stringToUTF8(str, ret, len);
        }
        return ret;
      },
      'array': (arr) {
        var ret = stackAlloc((arr as List).length);
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
            if (stack == 0) stack = stackSave();
            cArgs[i] = converter(args[i]);
          } else {
            cArgs[i] = args[i];
          }
        }
      }
      var ret = Function.apply(func, cArgs);
      ret = convertReturnValue(ret);
      if (stack != 0) stackRestore(stack);
      return ret;
    }

    Function cwrap(
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
      return varArgsFunction((arguments, _) {
        return ccall(ident, returnType, argTypes!, arguments, null);
      });
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
        preMain();
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
  var endIdx = maxBytesToRead != null ? idx + maxBytesToRead : double.maxFinite;
  var endPtr = idx;
  while (heap[endPtr] != 0 && !(endPtr >= endIdx)) {
    ++endPtr;
  }
  if (endPtr - idx > 16) {
    return utf8.decode(heap.sublist(idx, endPtr)); // TODO: view
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
    abort();
  }

  _emscripten_memcpy_big(int dest, int src, int num) {
    // TODO: HEAPU8.copyWithin(dest, src, src + num);
    final HEAPU8 = wasmMemory().view;
    // HEAPU8.setAll(dest, HEAPU8.sublist(src, src + num));
    List.copyRange(HEAPU8, dest, HEAPU8, src, src + num);
  }

  _emscripten_get_heap_size() {
    return wasmMemory().view.length;
  }

  emscripten_realloc_buffer(int size) {
    return reallocBuffer(size);
  }

  _emscripten_resize_heap(int requestedSize) {
    final oldSize = _emscripten_get_heap_size();
    const maxHeapSize = 2147483648;
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

  final SYSCALLSbuffers = [null, [], []];
  SYSCALLSprintChar(int stream, int curr) {
    final buffer = SYSCALLSbuffers[stream]! as Uint8List;
    if (curr == 0 || curr == 10) {
      printMsg(UTF8ArrayToString(buffer, 0, null), isError: stream != 1);
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
      var ptr = HEAP32[(iov + i * 8) >> 2];
      var len = HEAP32[(iov + (i * 8 + 4)) >> 2];
      for (var j = 0; j < len; j++) {
        SYSCALLSprintChar(fd, wasmMemory().view[ptr + j]);
      }
      num += len;
    }
    HEAP32[pnum >> 2] = num;
    return 0;
  }

  _pthread_create() {
    return 6;
  }

  _pthread_join() {
    return 28;
  }

  setErrNo(int value) {
    Int32List.view(wasmMemory().view.buffer)[___errno_location() >> 2] = value;
    return value;
  }

  _sysconf(name) {
    switch (name) {
      case 30:
        return 16384;
      case 85:
        var maxHeapSize = 2147483648;
        return maxHeapSize / 16384;
      case 132:
      case 133:
      case 12:
      case 137:
      case 138:
      case 15:
      case 235:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 149:
      case 13:
      case 10:
      case 236:
      case 153:
      case 9:
      case 21:
      case 22:
      case 159:
      case 154:
      case 14:
      case 77:
      case 78:
      case 139:
      case 82:
      case 68:
      case 67:
      case 164:
      case 11:
      case 29:
      case 47:
      case 48:
      case 95:
      case 52:
      case 51:
      case 46:
        return 200809;
      case 27:
      case 246:
      case 127:
      case 128:
      case 23:
      case 24:
      case 160:
      case 161:
      case 181:
      case 182:
      case 242:
      case 183:
      case 184:
      case 243:
      case 244:
      case 245:
      case 165:
      case 178:
      case 179:
      case 49:
      case 50:
      case 168:
      case 169:
      case 175:
      case 170:
      case 171:
      case 172:
      case 97:
      case 76:
      case 32:
      case 173:
      case 35:
      case 80:
      case 81:
      case 79:
        return -1;
      case 176:
      case 177:
      case 7:
      case 155:
      case 8:
      case 157:
      case 125:
      case 126:
      case 92:
      case 93:
      case 129:
      case 130:
      case 131:
      case 94:
      case 91:
        return 1;
      case 74:
      case 60:
      case 69:
      case 70:
      case 4:
        return 1024;
      case 31:
      case 42:
      case 72:
        return 32;
      case 87:
      case 26:
      case 33:
        return 2147483647;
      case 34:
      case 1:
        return 47839;
      case 38:
      case 36:
        return 99;
      case 43:
      case 37:
        return 2048;
      case 0:
        return 2097152;
      case 3:
        return 65536;
      case 28:
        return 32768;
      case 44:
        return 32767;
      case 75:
        return 16384;
      case 39:
        return 1e3;
      case 89:
        return 700;
      case 71:
        return 256;
      case 40:
        return 255;
      case 2:
        return 100;
      case 180:
        return 64;
      case 25:
        return 20;
      case 5:
        return 16;
      case 6:
        return 6;
      case 73:
        return 4;
      case 84:
        {
          try {
            return html.window.navigator.hardwareConcurrency ?? 1;
          } catch (_) {
            return 1;
          }
        }
    }
    setErrNo(28);
    return -1;
  }

  return {
    'a': _abort,
    'd': _emscripten_memcpy_big,
    'e': _emscripten_resize_heap,
    'f': _fd_close,
    'c': _fd_seek,
    'b': _fd_write,
    'h': _pthread_create,
    'g': _pthread_join,
    'i': _sysconf,
  };
}

void addTensorFlowFunctions(Map Module) async {
  for (final entry in tfNames.entries) {
    Module[entry.key] = varArgsFunction(
      (args, _) => Function.apply(
        (Module[entry.key] = Module['asm'][entry.value]) as Function,
        args,
      ),
    );
  }
}

const tfNames = {
  '_Abs': 't',
  '_Add': 's',
  '_AddN': 'u',
  '_All': 'v',
  '_Any': 'w',
  '_ArgMax': 'x',
  '_AvgPool': 'y',
  '_BatchMatMul': 'z',
  '_Ceil': 'A',
  '_ClipByValue': 'B',
  '_Conv2D': 'C',
  '_Conv2DBackpropInput': 'D',
  '_Cos': 'E',
  '_Cosh': 'F',
  '_CropAndResize': 'G',
  '_Cumsum': 'H',
  '_DepthToSpace': 'I',
  '_DepthwiseConv2dNative': 'J',
  '_Elu': 'K',
  '_Equal': 'L',
  '_Exp': 'M',
  '_FlipLeftRight': 'N',
  '_Floor': 'O',
  '_FloorDiv': 'P',
  '_FusedBatchNorm': 'Q',
  '_FusedConv2D': 'R',
  '_FusedDepthwiseConv2D': 'S',
  '_Gather': 'T',
  '_GatherNd': 'U',
  '_Greater': 'V',
  '_GreaterEqual': 'W',
  '_LeakyRelu': 'X',
  '_Less': 'Y',
  '_LessEqual': 'Z',
  '_Log': '_',
  '_LogicalAnd': r'$',
  '_Max': 'aa',
  '_MaxPool': 'ba',
  '_Maximum': 'ca',
  '_Mean': 'da',
  '_Min': 'ea',
  '_Minimum': 'fa',
  '_MirrorPad': 'ga',
  '_Multiply': 'ha',
  '_Neg': 'ia',
  '_NonMaxSuppressionV3': 'ja',
  '_NonMaxSuppressionV4': 'ka',
  '_NonMaxSuppressionV5': 'la',
  '_NotEqual': 'ma',
  '_OneHot': 'na',
  '_PadV2': 'oa',
  '_Pow': 'pa',
  '_Prelu': 'qa',
  '_Prod': 'ra',
  '_RealDiv': 'sa',
  '_Relu': 'ta',
  '_Relu6': 'ua',
  '_ResizeBilinear': 'va',
  '_Reverse': 'wa',
  '_RotateWithOffset': 'xa',
  '_Round': 'ya',
  '_Rsqrt': 'za',
  '_ScatterNd': 'Aa',
  '_SelectV2': 'Ba',
  '_Sigmoid': 'Ca',
  '_Sin': 'Da',
  '_Softmax': 'Ea',
  '_SparseFillEmptyRows': 'Fa',
  '_SparseReshape': 'Ga',
  '_SparseSegmentReduction': 'Ha',
  '_Sqrt': 'Ia',
  '_Square': 'Ja',
  '_SquaredDifference': 'Ka',
  '_Step': 'La',
  '_StridedSlice': 'Ma',
  '_Sub': 'Na',
  '_Sum': 'Oa',
  '_Tan': 'Pa',
  '_Tanh': 'Qa',
  '_Tile': 'Ra',
  '_TopK': 'Sa',
  '_Transform': 'Ta',
  '_Transpose': 'Ua',
  '__FusedMatMul': 'Va'
};

class WasmFactoryConfig {
  final Object? mainScriptUrlOrBlob; //: string|Blob;
  final String Function(String path, String prefix)? locateFile;
  final Function(
    Map<String, Map<String, Function>> info,
    void Function(WasmInstance, WasmModule),
  )? instantiateWasm;
  final void Function()? onRuntimeInitialized;
  final void Function(String)? onAbort;
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

typedef VarArgsCallback<T> = T Function(
    List<dynamic> args, Map<String, Object?> kwargs);

Function varArgsFunction<T>(VarArgsCallback<T> callback) {
  return VarArgsFunction._(callback);
}

class VarArgsFunction<T> {
  final VarArgsCallback<T> callback;
  static final _symbolOffset = 'Symbol("'.length;

  VarArgsFunction._(this.callback);

  T call() => callback([], {});

  @override
  dynamic noSuchMethod(Invocation inv) {
    return callback(
      inv.positionalArguments,
      inv.namedArguments.map(
        (_k, v) {
          var k = _k.toString();
          return MapEntry(k.substring(_symbolOffset, k.length - 2), v);
        },
      ),
    );
  }
}
