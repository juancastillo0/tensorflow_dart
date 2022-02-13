var WasmBackendModule = (function () {
  var _scriptDir =
    typeof document !== "undefined" && document.currentScript
      ? document.currentScript.src
      : undefined;
  if (typeof __filename !== "undefined") _scriptDir = _scriptDir || __filename;
  return function (WasmBackendModule) {
    WasmBackendModule = WasmBackendModule || {};

    var Module =
      typeof WasmBackendModule !== "undefined" ? WasmBackendModule : {};
    var readyPromiseResolve, readyPromiseReject;
    Module["ready"] = new Promise(function (resolve, reject) {
      readyPromiseResolve = resolve;
      readyPromiseReject = reject;
    });
    var beforeListeners;
    if (typeof process !== "undefined" && process.listeners) {
      beforeListeners = {
        uncaughtException: process.listeners("uncaughtException"),
        unhandledRejection: process.listeners("unhandledRejection"),
      };
    }
    var moduleOverrides = {};
    var key;
    for (key in Module) {
      if (Module.hasOwnProperty(key)) {
        moduleOverrides[key] = Module[key];
      }
    }
    var arguments_ = [];
    var thisProgram = "./this.program";
    var quit_ = function (status, toThrow) {
      throw toThrow;
    };
    var ENVIRONMENT_IS_WEB = false;
    var ENVIRONMENT_IS_WORKER = false;
    var ENVIRONMENT_IS_NODE = false;
    var ENVIRONMENT_IS_SHELL = false;
    ENVIRONMENT_IS_WEB = typeof window === "object";
    ENVIRONMENT_IS_WORKER = typeof importScripts === "function";
    ENVIRONMENT_IS_NODE =
      typeof process === "object" &&
      typeof process.versions === "object" &&
      typeof process.versions.node === "string";
    ENVIRONMENT_IS_SHELL =
      !ENVIRONMENT_IS_WEB && !ENVIRONMENT_IS_NODE && !ENVIRONMENT_IS_WORKER;
    var scriptDirectory = "";
    function locateFile(path) {
      if (Module["locateFile"]) {
        return Module["locateFile"](path, scriptDirectory);
      }
      return scriptDirectory + path;
    }
    var read_, readAsync, readBinary, setWindowTitle;
    var nodeFS;
    var nodePath;
    if (ENVIRONMENT_IS_NODE) {
      if (ENVIRONMENT_IS_WORKER) {
        scriptDirectory = require("path").dirname(scriptDirectory) + "/";
      } else {
        scriptDirectory = __dirname + "/";
      }
      read_ = function shell_read(filename, binary) {
        if (!nodeFS) nodeFS = require("fs");
        if (!nodePath) nodePath = require("path");
        filename = nodePath["normalize"](filename);
        return nodeFS["readFileSync"](filename, binary ? null : "utf8");
      };
      readBinary = function readBinary(filename) {
        var ret = read_(filename, true);
        if (!ret.buffer) {
          ret = new Uint8Array(ret);
        }
        assert(ret.buffer);
        return ret;
      };
      if (process["argv"].length > 1) {
        thisProgram = process["argv"][1].replace(/\\/g, "/");
      }
      arguments_ = process["argv"].slice(2);
      process["on"]("uncaughtException", function (ex) {
        if (!(ex instanceof ExitStatus)) {
          throw ex;
        }
      });
      process["on"]("unhandledRejection", abort);
      quit_ = function (status) {
        process["exit"](status);
      };
      Module["inspect"] = function () {
        return "[Emscripten Module object]";
      };
    } else if (ENVIRONMENT_IS_SHELL) {
      if (typeof read != "undefined") {
        read_ = function shell_read(f) {
          return read(f);
        };
      }
      readBinary = function readBinary(f) {
        var data;
        if (typeof readbuffer === "function") {
          return new Uint8Array(readbuffer(f));
        }
        data = read(f, "binary");
        assert(typeof data === "object");
        return data;
      };
      if (typeof scriptArgs != "undefined") {
        arguments_ = scriptArgs;
      } else if (typeof arguments != "undefined") {
        arguments_ = arguments;
      }
      if (typeof quit === "function") {
        quit_ = function (status) {
          quit(status);
        };
      }
      if (typeof print !== "undefined") {
        if (typeof console === "undefined") console = {};
        console.log = print;
        console.warn = console.error =
          typeof printErr !== "undefined" ? printErr : print;
      }
    } else if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
      if (ENVIRONMENT_IS_WORKER) {
        scriptDirectory = self.location.href;
      } else if (typeof document !== "undefined" && document.currentScript) {
        scriptDirectory = document.currentScript.src;
      }
      if (_scriptDir) {
        scriptDirectory = _scriptDir;
      }
      if (scriptDirectory.indexOf("blob:") !== 0) {
        scriptDirectory = scriptDirectory.substr(
          0,
          scriptDirectory.lastIndexOf("/") + 1
        );
      } else {
        scriptDirectory = "";
      }
      {
        read_ = function (url) {
          var xhr = new XMLHttpRequest();
          xhr.open("GET", url, false);
          xhr.send(null);
          return xhr.responseText;
        };
        if (ENVIRONMENT_IS_WORKER) {
          readBinary = function (url) {
            var xhr = new XMLHttpRequest();
            xhr.open("GET", url, false);
            xhr.responseType = "arraybuffer";
            xhr.send(null);
            return new Uint8Array(xhr.response);
          };
        }
        readAsync = function (url, onload, onerror) {
          var xhr = new XMLHttpRequest();
          xhr.open("GET", url, true);
          xhr.responseType = "arraybuffer";
          xhr.onload = function () {
            if (xhr.status == 200 || (xhr.status == 0 && xhr.response)) {
              onload(xhr.response);
              return;
            }
            onerror();
          };
          xhr.onerror = onerror;
          xhr.send(null);
        };
      }
      setWindowTitle = function (title) {
        document.title = title;
      };
    } else {
    }
    var out = Module["print"] || console.log.bind(console);
    var err = Module["printErr"] || console.warn.bind(console);
    for (key in moduleOverrides) {
      if (moduleOverrides.hasOwnProperty(key)) {
        Module[key] = moduleOverrides[key];
      }
    }
    moduleOverrides = null;
    if (Module["arguments"]) arguments_ = Module["arguments"];
    if (Module["thisProgram"]) thisProgram = Module["thisProgram"];
    if (Module["quit"]) quit_ = Module["quit"];
    var wasmBinary;
    if (Module["wasmBinary"]) wasmBinary = Module["wasmBinary"];
    var noExitRuntime = Module["noExitRuntime"] || true;
    if (typeof WebAssembly !== "object") {
      abort("no native wasm support detected");
    }
    var wasmMemory;
    var ABORT = false;
    var EXITSTATUS;
    function assert(condition, text) {
      if (!condition) {
        abort("Assertion failed: " + text);
      }
    }
    function getCFunc(ident) {
      var func = Module["_" + ident];
      assert(
        func,
        "Cannot call unknown function " + ident + ", make sure it is exported"
      );
      return func;
    }
    function ccall(ident, returnType, argTypes, args, opts) {
      var toC = {
        string: function (str) {
          var ret = 0;
          if (str !== null && str !== undefined && str !== 0) {
            var len = (str.length << 2) + 1;
            ret = stackAlloc(len);
            stringToUTF8(str, ret, len);
          }
          return ret;
        },
        array: function (arr) {
          var ret = stackAlloc(arr.length);
          writeArrayToMemory(arr, ret);
          return ret;
        },
      };
      function convertReturnValue(ret) {
        if (returnType === "string") return UTF8ToString(ret);
        if (returnType === "boolean") return Boolean(ret);
        return ret;
      }
      var func = getCFunc(ident);
      var cArgs = [];
      var stack = 0;
      if (args) {
        for (var i = 0; i < args.length; i++) {
          var converter = toC[argTypes[i]];
          if (converter) {
            if (stack === 0) stack = stackSave();
            cArgs[i] = converter(args[i]);
          } else {
            cArgs[i] = args[i];
          }
        }
      }
      var ret = func.apply(null, cArgs);
      ret = convertReturnValue(ret);
      if (stack !== 0) stackRestore(stack);
      return ret;
    }
    function cwrap(ident, returnType, argTypes, opts) {
      argTypes = argTypes || [];
      var numericArgs = argTypes.every(function (type) {
        return type === "number";
      });
      var numericRet = returnType !== "string";
      if (numericRet && numericArgs && !opts) {
        return getCFunc(ident);
      }
      return function () {
        return ccall(ident, returnType, argTypes, arguments, opts);
      };
    }
    var UTF8Decoder =
      typeof TextDecoder !== "undefined" ? new TextDecoder("utf8") : undefined;
    function UTF8ArrayToString(heap, idx, maxBytesToRead) {
      var endIdx = idx + maxBytesToRead;
      var endPtr = idx;
      while (heap[endPtr] && !(endPtr >= endIdx)) ++endPtr;
      if (endPtr - idx > 16 && heap.subarray && UTF8Decoder) {
        return UTF8Decoder.decode(heap.subarray(idx, endPtr));
      } else {
        var str = "";
        while (idx < endPtr) {
          var u0 = heap[idx++];
          if (!(u0 & 128)) {
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
            str += String.fromCharCode(55296 | (ch >> 10), 56320 | (ch & 1023));
          }
        }
      }
      return str;
    }
    function UTF8ToString(ptr, maxBytesToRead) {
      return ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead) : "";
    }
    function stringToUTF8Array(str, heap, outIdx, maxBytesToWrite) {
      if (!(maxBytesToWrite > 0)) return 0;
      var startIdx = outIdx;
      var endIdx = outIdx + maxBytesToWrite - 1;
      for (var i = 0; i < str.length; ++i) {
        var u = str.charCodeAt(i);
        if (u >= 55296 && u <= 57343) {
          var u1 = str.charCodeAt(++i);
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
    function stringToUTF8(str, outPtr, maxBytesToWrite) {
      return stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite);
    }
    function writeArrayToMemory(array, buffer) {
      HEAP8.set(array, buffer);
    }
    function alignUp(x, multiple) {
      if (x % multiple > 0) {
        x += multiple - (x % multiple);
      }
      return x;
    }
    var buffer,
      HEAP8,
      HEAPU8,
      HEAP16,
      HEAPU16,
      HEAP32,
      HEAPU32,
      HEAPF32,
      HEAPF64;
    function updateGlobalBufferAndViews(buf) {
      buffer = buf;
      Module["HEAP8"] = HEAP8 = new Int8Array(buf);
      Module["HEAP16"] = HEAP16 = new Int16Array(buf);
      Module["HEAP32"] = HEAP32 = new Int32Array(buf);
      Module["HEAPU8"] = HEAPU8 = new Uint8Array(buf);
      Module["HEAPU16"] = HEAPU16 = new Uint16Array(buf);
      Module["HEAPU32"] = HEAPU32 = new Uint32Array(buf);
      Module["HEAPF32"] = HEAPF32 = new Float32Array(buf);
      Module["HEAPF64"] = HEAPF64 = new Float64Array(buf);
    }
    var INITIAL_MEMORY = Module["INITIAL_MEMORY"] || 16777216;
    var wasmTable;
    var __ATPRERUN__ = [];
    var __ATINIT__ = [];
    var __ATMAIN__ = [];
    var __ATPOSTRUN__ = [];
    var runtimeInitialized = false;
    __ATINIT__.push({
      func: function () {
        ___wasm_call_ctors();
      },
    });
    function preRun() {
      if (Module["preRun"]) {
        if (typeof Module["preRun"] == "function")
          Module["preRun"] = [Module["preRun"]];
        while (Module["preRun"].length) {
          addOnPreRun(Module["preRun"].shift());
        }
      }
      callRuntimeCallbacks(__ATPRERUN__);
    }
    function initRuntime() {
      runtimeInitialized = true;
      callRuntimeCallbacks(__ATINIT__);
    }
    function preMain() {
      callRuntimeCallbacks(__ATMAIN__);
    }
    function postRun() {
      if (Module["postRun"]) {
        if (typeof Module["postRun"] == "function")
          Module["postRun"] = [Module["postRun"]];
        while (Module["postRun"].length) {
          addOnPostRun(Module["postRun"].shift());
        }
      }
      callRuntimeCallbacks(__ATPOSTRUN__);
    }
    function addOnPreRun(cb) {
      __ATPRERUN__.unshift(cb);
    }
    function addOnPostRun(cb) {
      __ATPOSTRUN__.unshift(cb);
    }
    var runDependencies = 0;
    var runDependencyWatcher = null;
    var dependenciesFulfilled = null;
    function addRunDependency(id) {
      runDependencies++;
      if (Module["monitorRunDependencies"]) {
        Module["monitorRunDependencies"](runDependencies);
      }
    }
    function removeRunDependency(id) {
      runDependencies--;
      if (Module["monitorRunDependencies"]) {
        Module["monitorRunDependencies"](runDependencies);
      }
      if (runDependencies == 0) {
        if (runDependencyWatcher !== null) {
          clearInterval(runDependencyWatcher);
          runDependencyWatcher = null;
        }
        if (dependenciesFulfilled) {
          var callback = dependenciesFulfilled;
          dependenciesFulfilled = null;
          callback();
        }
      }
    }
    Module["preloadedImages"] = {};
    Module["preloadedAudios"] = {};
    function abort(what) {
      if (Module["onAbort"]) {
        Module["onAbort"](what);
      }
      what += "";
      err(what);
      ABORT = true;
      EXITSTATUS = 1;
      what = "abort(" + what + "). Build with -s ASSERTIONS=1 for more info.";
      var e = new WebAssembly.RuntimeError(what);
      readyPromiseReject(e);
      throw e;
    }
    function hasPrefix(str, prefix) {
      return String.prototype.startsWith
        ? str.startsWith(prefix)
        : str.indexOf(prefix) === 0;
    }
    var dataURIPrefix = "data:application/octet-stream;base64,";
    function isDataURI(filename) {
      return hasPrefix(filename, dataURIPrefix);
    }
    var fileURIPrefix = "file://";
    function isFileURI(filename) {
      return hasPrefix(filename, fileURIPrefix);
    }
    var wasmBinaryFile = "tfjs-backend-wasm.wasm";
    if (!isDataURI(wasmBinaryFile)) {
      wasmBinaryFile = locateFile(wasmBinaryFile);
    }
    function getBinary(file) {
      try {
        if (file == wasmBinaryFile && wasmBinary) {
          return new Uint8Array(wasmBinary);
        }
        if (readBinary) {
          return readBinary(file);
        } else {
          throw "both async and sync fetching of the wasm failed";
        }
      } catch (err) {
        abort(err);
      }
    }
    function getBinaryPromise() {
      if (!wasmBinary && (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER)) {
        if (typeof fetch === "function" && !isFileURI(wasmBinaryFile)) {
          return fetch(wasmBinaryFile, { credentials: "same-origin" })
            .then(function (response) {
              if (!response["ok"]) {
                throw (
                  "failed to load wasm binary file at '" + wasmBinaryFile + "'"
                );
              }
              return response["arrayBuffer"]();
            })
            .catch(function () {
              return getBinary(wasmBinaryFile);
            });
        } else {
          if (readAsync) {
            return new Promise(function (resolve, reject) {
              readAsync(
                wasmBinaryFile,
                function (response) {
                  resolve(new Uint8Array(response));
                },
                reject
              );
            });
          }
        }
      }
      return Promise.resolve().then(function () {
        return getBinary(wasmBinaryFile);
      });
    }
    function createWasm() {
      var info = { a: asmLibraryArg };
      function receiveInstance(instance, module) {
        var exports = instance.exports;
        Module["asm"] = exports;
        wasmMemory = Module["asm"]["j"];
        updateGlobalBufferAndViews(wasmMemory.buffer);
        wasmTable = Module["asm"]["r"];
        removeRunDependency("wasm-instantiate");
      }
      addRunDependency("wasm-instantiate");
      function receiveInstantiatedSource(output) {
        receiveInstance(output["instance"]);
      }
      function instantiateArrayBuffer(receiver) {
        return getBinaryPromise()
          .then(function (binary) {
            return WebAssembly.instantiate(binary, info);
          })
          .then(receiver, function (reason) {
            err("failed to asynchronously prepare wasm: " + reason);
            abort(reason);
          });
      }
      function instantiateAsync() {
        if (
          !wasmBinary &&
          typeof WebAssembly.instantiateStreaming === "function" &&
          !isDataURI(wasmBinaryFile) &&
          !isFileURI(wasmBinaryFile) &&
          typeof fetch === "function"
        ) {
          return fetch(wasmBinaryFile, { credentials: "same-origin" }).then(
            function (response) {
              var result = WebAssembly.instantiateStreaming(response, info);
              return result.then(receiveInstantiatedSource, function (reason) {
                err("wasm streaming compile failed: " + reason);
                err("falling back to ArrayBuffer instantiation");
                return instantiateArrayBuffer(receiveInstantiatedSource);
              });
            }
          );
        } else {
          return instantiateArrayBuffer(receiveInstantiatedSource);
        }
      }
      if (Module["instantiateWasm"]) {
        try {
          var exports = Module["instantiateWasm"](info, receiveInstance);
          return exports;
        } catch (e) {
          err("Module.instantiateWasm callback failed with error: " + e);
          return false;
        }
      }
      instantiateAsync().catch(readyPromiseReject);
      return {};
    }
    function callRuntimeCallbacks(callbacks) {
      while (callbacks.length > 0) {
        var callback = callbacks.shift();
        if (typeof callback == "function") {
          callback(Module);
          continue;
        }
        var func = callback.func;
        if (typeof func === "number") {
          if (callback.arg === undefined) {
            wasmTable.get(func)();
          } else {
            wasmTable.get(func)(callback.arg);
          }
        } else {
          func(callback.arg === undefined ? null : callback.arg);
        }
      }
    }
    function _abort() {
      abort();
    }
    function _emscripten_memcpy_big(dest, src, num) {
      HEAPU8.copyWithin(dest, src, src + num);
    }
    function _emscripten_get_heap_size() {
      return HEAPU8.length;
    }
    function emscripten_realloc_buffer(size) {
      try {
        wasmMemory.grow((size - buffer.byteLength + 65535) >>> 16);
        updateGlobalBufferAndViews(wasmMemory.buffer);
        return 1;
      } catch (e) {}
    }
    function _emscripten_resize_heap(requestedSize) {
      var oldSize = _emscripten_get_heap_size();
      var maxHeapSize = 2147483648;
      if (requestedSize > maxHeapSize) {
        return false;
      }
      for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
        var overGrownHeapSize = oldSize * (1 + 0.2 / cutDown);
        overGrownHeapSize = Math.min(
          overGrownHeapSize,
          requestedSize + 100663296
        );
        var newSize = Math.min(
          maxHeapSize,
          alignUp(Math.max(requestedSize, overGrownHeapSize), 65536)
        );
        var replacement = emscripten_realloc_buffer(newSize);
        if (replacement) {
          return true;
        }
      }
      return false;
    }
    var SYSCALLS = {
      mappings: {},
      buffers: [null, [], []],
      printChar: function (stream, curr) {
        var buffer = SYSCALLS.buffers[stream];
        if (curr === 0 || curr === 10) {
          (stream === 1 ? out : err)(UTF8ArrayToString(buffer, 0));
          buffer.length = 0;
        } else {
          buffer.push(curr);
        }
      },
      varargs: undefined,
      get: function () {
        SYSCALLS.varargs += 4;
        var ret = HEAP32[(SYSCALLS.varargs - 4) >> 2];
        return ret;
      },
      getStr: function (ptr) {
        var ret = UTF8ToString(ptr);
        return ret;
      },
      get64: function (low, high) {
        return low;
      },
    };
    function _fd_close(fd) {
      return 0;
    }
    function _fd_seek(fd, offset_low, offset_high, whence, newOffset) {}
    function _fd_write(fd, iov, iovcnt, pnum) {
      var num = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAP32[(iov + i * 8) >> 2];
        var len = HEAP32[(iov + (i * 8 + 4)) >> 2];
        for (var j = 0; j < len; j++) {
          SYSCALLS.printChar(fd, HEAPU8[ptr + j]);
        }
        num += len;
      }
      HEAP32[pnum >> 2] = num;
      return 0;
    }
    function _pthread_create() {
      return 6;
    }
    function _pthread_join() {
      return 28;
    }
    function setErrNo(value) {
      HEAP32[___errno_location() >> 2] = value;
      return value;
    }
    function _sysconf(name) {
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
        case 84: {
          if (typeof navigator === "object")
            return navigator["hardwareConcurrency"] || 1;
          return 1;
        }
      }
      setErrNo(28);
      return -1;
    }
    var asmLibraryArg = {
      a: _abort,
      d: _emscripten_memcpy_big,
      e: _emscripten_resize_heap,
      f: _fd_close,
      c: _fd_seek,
      b: _fd_write,
      h: _pthread_create,
      g: _pthread_join,
      i: _sysconf,
    };
    var asm = createWasm();
    var ___wasm_call_ctors = (Module["___wasm_call_ctors"] = function () {
      return (___wasm_call_ctors = Module["___wasm_call_ctors"] =
        Module["asm"]["k"]).apply(null, arguments);
    });
    var _init = (Module["_init"] = function () {
      return (_init = Module["_init"] = Module["asm"]["l"]).apply(
        null,
        arguments
      );
    });
    var _init_with_threads_count = (Module["_init_with_threads_count"] =
      function () {
        return (_init_with_threads_count = Module["_init_with_threads_count"] =
          Module["asm"]["m"]).apply(null, arguments);
      });
    var _get_threads_count = (Module["_get_threads_count"] = function () {
      return (_get_threads_count = Module["_get_threads_count"] =
        Module["asm"]["n"]).apply(null, arguments);
    });
    var _register_tensor = (Module["_register_tensor"] = function () {
      return (_register_tensor = Module["_register_tensor"] =
        Module["asm"]["o"]).apply(null, arguments);
    });
    var _dispose_data = (Module["_dispose_data"] = function () {
      return (_dispose_data = Module["_dispose_data"] =
        Module["asm"]["p"]).apply(null, arguments);
    });
    var _dispose = (Module["_dispose"] = function () {
      return (_dispose = Module["_dispose"] = Module["asm"]["q"]).apply(
        null,
        arguments
      );
    });
    var _Abs = (Module["_Abs"] = function () {
      return (_Abs = Module["_Abs"] = Module["asm"]["s"]).apply(
        null,
        arguments
      );
    });
    var _Add = (Module["_Add"] = function () {
      return (_Add = Module["_Add"] = Module["asm"]["t"]).apply(
        null,
        arguments
      );
    });
    var _AddN = (Module["_AddN"] = function () {
      return (_AddN = Module["_AddN"] = Module["asm"]["u"]).apply(
        null,
        arguments
      );
    });
    var _All = (Module["_All"] = function () {
      return (_All = Module["_All"] = Module["asm"]["v"]).apply(
        null,
        arguments
      );
    });
    var _Any = (Module["_Any"] = function () {
      return (_Any = Module["_Any"] = Module["asm"]["w"]).apply(
        null,
        arguments
      );
    });
    var _ArgMax = (Module["_ArgMax"] = function () {
      return (_ArgMax = Module["_ArgMax"] = Module["asm"]["x"]).apply(
        null,
        arguments
      );
    });
    var _AvgPool = (Module["_AvgPool"] = function () {
      return (_AvgPool = Module["_AvgPool"] = Module["asm"]["y"]).apply(
        null,
        arguments
      );
    });
    var _BatchMatMul = (Module["_BatchMatMul"] = function () {
      return (_BatchMatMul = Module["_BatchMatMul"] = Module["asm"]["z"]).apply(
        null,
        arguments
      );
    });
    var _Ceil = (Module["_Ceil"] = function () {
      return (_Ceil = Module["_Ceil"] = Module["asm"]["A"]).apply(
        null,
        arguments
      );
    });
    var _ClipByValue = (Module["_ClipByValue"] = function () {
      return (_ClipByValue = Module["_ClipByValue"] = Module["asm"]["B"]).apply(
        null,
        arguments
      );
    });
    var _Conv2D = (Module["_Conv2D"] = function () {
      return (_Conv2D = Module["_Conv2D"] = Module["asm"]["C"]).apply(
        null,
        arguments
      );
    });
    var _Conv2DBackpropInput = (Module["_Conv2DBackpropInput"] = function () {
      return (_Conv2DBackpropInput = Module["_Conv2DBackpropInput"] =
        Module["asm"]["D"]).apply(null, arguments);
    });
    var _Cos = (Module["_Cos"] = function () {
      return (_Cos = Module["_Cos"] = Module["asm"]["E"]).apply(
        null,
        arguments
      );
    });
    var _Cosh = (Module["_Cosh"] = function () {
      return (_Cosh = Module["_Cosh"] = Module["asm"]["F"]).apply(
        null,
        arguments
      );
    });
    var _CropAndResize = (Module["_CropAndResize"] = function () {
      return (_CropAndResize = Module["_CropAndResize"] =
        Module["asm"]["G"]).apply(null, arguments);
    });
    var _Cumsum = (Module["_Cumsum"] = function () {
      return (_Cumsum = Module["_Cumsum"] = Module["asm"]["H"]).apply(
        null,
        arguments
      );
    });
    var _DepthToSpace = (Module["_DepthToSpace"] = function () {
      return (_DepthToSpace = Module["_DepthToSpace"] =
        Module["asm"]["I"]).apply(null, arguments);
    });
    var _DepthwiseConv2dNative = (Module["_DepthwiseConv2dNative"] =
      function () {
        return (_DepthwiseConv2dNative = Module["_DepthwiseConv2dNative"] =
          Module["asm"]["J"]).apply(null, arguments);
      });
    var _Elu = (Module["_Elu"] = function () {
      return (_Elu = Module["_Elu"] = Module["asm"]["K"]).apply(
        null,
        arguments
      );
    });
    var _Equal = (Module["_Equal"] = function () {
      return (_Equal = Module["_Equal"] = Module["asm"]["L"]).apply(
        null,
        arguments
      );
    });
    var _Exp = (Module["_Exp"] = function () {
      return (_Exp = Module["_Exp"] = Module["asm"]["M"]).apply(
        null,
        arguments
      );
    });
    var _FlipLeftRight = (Module["_FlipLeftRight"] = function () {
      return (_FlipLeftRight = Module["_FlipLeftRight"] =
        Module["asm"]["N"]).apply(null, arguments);
    });
    var _Floor = (Module["_Floor"] = function () {
      return (_Floor = Module["_Floor"] = Module["asm"]["O"]).apply(
        null,
        arguments
      );
    });
    var _FloorDiv = (Module["_FloorDiv"] = function () {
      return (_FloorDiv = Module["_FloorDiv"] = Module["asm"]["P"]).apply(
        null,
        arguments
      );
    });
    var _FusedBatchNorm = (Module["_FusedBatchNorm"] = function () {
      return (_FusedBatchNorm = Module["_FusedBatchNorm"] =
        Module["asm"]["Q"]).apply(null, arguments);
    });
    var _FusedConv2D = (Module["_FusedConv2D"] = function () {
      return (_FusedConv2D = Module["_FusedConv2D"] = Module["asm"]["R"]).apply(
        null,
        arguments
      );
    });
    var _FusedDepthwiseConv2D = (Module["_FusedDepthwiseConv2D"] = function () {
      return (_FusedDepthwiseConv2D = Module["_FusedDepthwiseConv2D"] =
        Module["asm"]["S"]).apply(null, arguments);
    });
    var _Gather = (Module["_Gather"] = function () {
      return (_Gather = Module["_Gather"] = Module["asm"]["T"]).apply(
        null,
        arguments
      );
    });
    var _GatherNd = (Module["_GatherNd"] = function () {
      return (_GatherNd = Module["_GatherNd"] = Module["asm"]["U"]).apply(
        null,
        arguments
      );
    });
    var _Greater = (Module["_Greater"] = function () {
      return (_Greater = Module["_Greater"] = Module["asm"]["V"]).apply(
        null,
        arguments
      );
    });
    var _GreaterEqual = (Module["_GreaterEqual"] = function () {
      return (_GreaterEqual = Module["_GreaterEqual"] =
        Module["asm"]["W"]).apply(null, arguments);
    });
    var _LeakyRelu = (Module["_LeakyRelu"] = function () {
      return (_LeakyRelu = Module["_LeakyRelu"] = Module["asm"]["X"]).apply(
        null,
        arguments
      );
    });
    var _Less = (Module["_Less"] = function () {
      return (_Less = Module["_Less"] = Module["asm"]["Y"]).apply(
        null,
        arguments
      );
    });
    var _LessEqual = (Module["_LessEqual"] = function () {
      return (_LessEqual = Module["_LessEqual"] = Module["asm"]["Z"]).apply(
        null,
        arguments
      );
    });
    var _Log = (Module["_Log"] = function () {
      return (_Log = Module["_Log"] = Module["asm"]["_"]).apply(
        null,
        arguments
      );
    });
    var _LogicalAnd = (Module["_LogicalAnd"] = function () {
      return (_LogicalAnd = Module["_LogicalAnd"] = Module["asm"]["$"]).apply(
        null,
        arguments
      );
    });
    var _Max = (Module["_Max"] = function () {
      return (_Max = Module["_Max"] = Module["asm"]["aa"]).apply(
        null,
        arguments
      );
    });
    var _MaxPool = (Module["_MaxPool"] = function () {
      return (_MaxPool = Module["_MaxPool"] = Module["asm"]["ba"]).apply(
        null,
        arguments
      );
    });
    var _Maximum = (Module["_Maximum"] = function () {
      return (_Maximum = Module["_Maximum"] = Module["asm"]["ca"]).apply(
        null,
        arguments
      );
    });
    var _Mean = (Module["_Mean"] = function () {
      return (_Mean = Module["_Mean"] = Module["asm"]["da"]).apply(
        null,
        arguments
      );
    });
    var _Min = (Module["_Min"] = function () {
      return (_Min = Module["_Min"] = Module["asm"]["ea"]).apply(
        null,
        arguments
      );
    });
    var _Minimum = (Module["_Minimum"] = function () {
      return (_Minimum = Module["_Minimum"] = Module["asm"]["fa"]).apply(
        null,
        arguments
      );
    });
    var _MirrorPad = (Module["_MirrorPad"] = function () {
      return (_MirrorPad = Module["_MirrorPad"] = Module["asm"]["ga"]).apply(
        null,
        arguments
      );
    });
    var _Multiply = (Module["_Multiply"] = function () {
      return (_Multiply = Module["_Multiply"] = Module["asm"]["ha"]).apply(
        null,
        arguments
      );
    });
    var _Neg = (Module["_Neg"] = function () {
      return (_Neg = Module["_Neg"] = Module["asm"]["ia"]).apply(
        null,
        arguments
      );
    });
    var _NonMaxSuppressionV3 = (Module["_NonMaxSuppressionV3"] = function () {
      return (_NonMaxSuppressionV3 = Module["_NonMaxSuppressionV3"] =
        Module["asm"]["ja"]).apply(null, arguments);
    });
    var _NonMaxSuppressionV4 = (Module["_NonMaxSuppressionV4"] = function () {
      return (_NonMaxSuppressionV4 = Module["_NonMaxSuppressionV4"] =
        Module["asm"]["ka"]).apply(null, arguments);
    });
    var _NonMaxSuppressionV5 = (Module["_NonMaxSuppressionV5"] = function () {
      return (_NonMaxSuppressionV5 = Module["_NonMaxSuppressionV5"] =
        Module["asm"]["la"]).apply(null, arguments);
    });
    var _NotEqual = (Module["_NotEqual"] = function () {
      return (_NotEqual = Module["_NotEqual"] = Module["asm"]["ma"]).apply(
        null,
        arguments
      );
    });
    var _OneHot = (Module["_OneHot"] = function () {
      return (_OneHot = Module["_OneHot"] = Module["asm"]["na"]).apply(
        null,
        arguments
      );
    });
    var _PadV2 = (Module["_PadV2"] = function () {
      return (_PadV2 = Module["_PadV2"] = Module["asm"]["oa"]).apply(
        null,
        arguments
      );
    });
    var _Pow = (Module["_Pow"] = function () {
      return (_Pow = Module["_Pow"] = Module["asm"]["pa"]).apply(
        null,
        arguments
      );
    });
    var _Prelu = (Module["_Prelu"] = function () {
      return (_Prelu = Module["_Prelu"] = Module["asm"]["qa"]).apply(
        null,
        arguments
      );
    });
    var _Prod = (Module["_Prod"] = function () {
      return (_Prod = Module["_Prod"] = Module["asm"]["ra"]).apply(
        null,
        arguments
      );
    });
    var _RealDiv = (Module["_RealDiv"] = function () {
      return (_RealDiv = Module["_RealDiv"] = Module["asm"]["sa"]).apply(
        null,
        arguments
      );
    });
    var _Relu = (Module["_Relu"] = function () {
      return (_Relu = Module["_Relu"] = Module["asm"]["ta"]).apply(
        null,
        arguments
      );
    });
    var _Relu6 = (Module["_Relu6"] = function () {
      return (_Relu6 = Module["_Relu6"] = Module["asm"]["ua"]).apply(
        null,
        arguments
      );
    });
    var _ResizeBilinear = (Module["_ResizeBilinear"] = function () {
      return (_ResizeBilinear = Module["_ResizeBilinear"] =
        Module["asm"]["va"]).apply(null, arguments);
    });
    var _Reverse = (Module["_Reverse"] = function () {
      return (_Reverse = Module["_Reverse"] = Module["asm"]["wa"]).apply(
        null,
        arguments
      );
    });
    var _RotateWithOffset = (Module["_RotateWithOffset"] = function () {
      return (_RotateWithOffset = Module["_RotateWithOffset"] =
        Module["asm"]["xa"]).apply(null, arguments);
    });
    var _Round = (Module["_Round"] = function () {
      return (_Round = Module["_Round"] = Module["asm"]["ya"]).apply(
        null,
        arguments
      );
    });
    var _Rsqrt = (Module["_Rsqrt"] = function () {
      return (_Rsqrt = Module["_Rsqrt"] = Module["asm"]["za"]).apply(
        null,
        arguments
      );
    });
    var _ScatterNd = (Module["_ScatterNd"] = function () {
      return (_ScatterNd = Module["_ScatterNd"] = Module["asm"]["Aa"]).apply(
        null,
        arguments
      );
    });
    var _SelectV2 = (Module["_SelectV2"] = function () {
      return (_SelectV2 = Module["_SelectV2"] = Module["asm"]["Ba"]).apply(
        null,
        arguments
      );
    });
    var _Sigmoid = (Module["_Sigmoid"] = function () {
      return (_Sigmoid = Module["_Sigmoid"] = Module["asm"]["Ca"]).apply(
        null,
        arguments
      );
    });
    var _Sin = (Module["_Sin"] = function () {
      return (_Sin = Module["_Sin"] = Module["asm"]["Da"]).apply(
        null,
        arguments
      );
    });
    var _Softmax = (Module["_Softmax"] = function () {
      return (_Softmax = Module["_Softmax"] = Module["asm"]["Ea"]).apply(
        null,
        arguments
      );
    });
    var _SparseFillEmptyRows = (Module["_SparseFillEmptyRows"] = function () {
      return (_SparseFillEmptyRows = Module["_SparseFillEmptyRows"] =
        Module["asm"]["Fa"]).apply(null, arguments);
    });
    var _SparseReshape = (Module["_SparseReshape"] = function () {
      return (_SparseReshape = Module["_SparseReshape"] =
        Module["asm"]["Ga"]).apply(null, arguments);
    });
    var _SparseSegmentReduction = (Module["_SparseSegmentReduction"] =
      function () {
        return (_SparseSegmentReduction = Module["_SparseSegmentReduction"] =
          Module["asm"]["Ha"]).apply(null, arguments);
      });
    var _Sqrt = (Module["_Sqrt"] = function () {
      return (_Sqrt = Module["_Sqrt"] = Module["asm"]["Ia"]).apply(
        null,
        arguments
      );
    });
    var _Square = (Module["_Square"] = function () {
      return (_Square = Module["_Square"] = Module["asm"]["Ja"]).apply(
        null,
        arguments
      );
    });
    var _SquaredDifference = (Module["_SquaredDifference"] = function () {
      return (_SquaredDifference = Module["_SquaredDifference"] =
        Module["asm"]["Ka"]).apply(null, arguments);
    });
    var _Step = (Module["_Step"] = function () {
      return (_Step = Module["_Step"] = Module["asm"]["La"]).apply(
        null,
        arguments
      );
    });
    var _StridedSlice = (Module["_StridedSlice"] = function () {
      return (_StridedSlice = Module["_StridedSlice"] =
        Module["asm"]["Ma"]).apply(null, arguments);
    });
    var _Sub = (Module["_Sub"] = function () {
      return (_Sub = Module["_Sub"] = Module["asm"]["Na"]).apply(
        null,
        arguments
      );
    });
    var _Sum = (Module["_Sum"] = function () {
      return (_Sum = Module["_Sum"] = Module["asm"]["Oa"]).apply(
        null,
        arguments
      );
    });
    var _Tan = (Module["_Tan"] = function () {
      return (_Tan = Module["_Tan"] = Module["asm"]["Pa"]).apply(
        null,
        arguments
      );
    });
    var _Tanh = (Module["_Tanh"] = function () {
      return (_Tanh = Module["_Tanh"] = Module["asm"]["Qa"]).apply(
        null,
        arguments
      );
    });
    var _Tile = (Module["_Tile"] = function () {
      return (_Tile = Module["_Tile"] = Module["asm"]["Ra"]).apply(
        null,
        arguments
      );
    });
    var _TopK = (Module["_TopK"] = function () {
      return (_TopK = Module["_TopK"] = Module["asm"]["Sa"]).apply(
        null,
        arguments
      );
    });
    var _Transform = (Module["_Transform"] = function () {
      return (_Transform = Module["_Transform"] = Module["asm"]["Ta"]).apply(
        null,
        arguments
      );
    });
    var _Transpose = (Module["_Transpose"] = function () {
      return (_Transpose = Module["_Transpose"] = Module["asm"]["Ua"]).apply(
        null,
        arguments
      );
    });
    var __FusedMatMul = (Module["__FusedMatMul"] = function () {
      return (__FusedMatMul = Module["__FusedMatMul"] =
        Module["asm"]["Va"]).apply(null, arguments);
    });
    var _malloc = (Module["_malloc"] = function () {
      return (_malloc = Module["_malloc"] = Module["asm"]["Wa"]).apply(
        null,
        arguments
      );
    });
    var _free = (Module["_free"] = function () {
      return (_free = Module["_free"] = Module["asm"]["Xa"]).apply(
        null,
        arguments
      );
    });
    var ___errno_location = (Module["___errno_location"] = function () {
      return (___errno_location = Module["___errno_location"] =
        Module["asm"]["Ya"]).apply(null, arguments);
    });
    var stackSave = (Module["stackSave"] = function () {
      return (stackSave = Module["stackSave"] = Module["asm"]["Za"]).apply(
        null,
        arguments
      );
    });
    var stackRestore = (Module["stackRestore"] = function () {
      return (stackRestore = Module["stackRestore"] =
        Module["asm"]["_a"]).apply(null, arguments);
    });
    var stackAlloc = (Module["stackAlloc"] = function () {
      return (stackAlloc = Module["stackAlloc"] = Module["asm"]["$a"]).apply(
        null,
        arguments
      );
    });
    Module["cwrap"] = cwrap;
    var calledRun;
    function ExitStatus(status) {
      this.name = "ExitStatus";
      this.message = "Program terminated with exit(" + status + ")";
      this.status = status;
    }
    dependenciesFulfilled = function runCaller() {
      if (!calledRun) run();
      if (!calledRun) dependenciesFulfilled = runCaller;
    };
    function run(args) {
      args = args || arguments_;
      if (runDependencies > 0) {
        return;
      }
      preRun();
      if (runDependencies > 0) {
        return;
      }
      function doRun() {
        if (calledRun) return;
        calledRun = true;
        Module["calledRun"] = true;
        if (ABORT) return;
        initRuntime();
        preMain();
        readyPromiseResolve(Module);
        if (Module["onRuntimeInitialized"]) Module["onRuntimeInitialized"]();
        postRun();
      }
      if (Module["setStatus"]) {
        Module["setStatus"]("Running...");
        setTimeout(function () {
          setTimeout(function () {
            Module["setStatus"]("");
          }, 1);
          doRun();
        }, 1);
      } else {
        doRun();
      }
    }
    Module["run"] = run;
    if (Module["preInit"]) {
      if (typeof Module["preInit"] == "function")
        Module["preInit"] = [Module["preInit"]];
      while (Module["preInit"].length > 0) {
        Module["preInit"].pop()();
      }
    }
    run();
    var listenersAdded;
    if (beforeListeners) {
      listenersAdded = {
        uncaughtException: process
          .listeners("uncaughtException")
          .filter(function (listener) {
            return !beforeListeners.uncaughtException.indexOf(listener) > -1;
          }),
        unhandledRejection: process
          .listeners("unhandledRejection")
          .filter(function (listener) {
            return !beforeListeners.unhandledRejection.indexOf(listener) > -1;
          }),
      };
    }
    var actualModule;
    if (typeof WasmBackendModule !== "undefined") {
      actualModule = WasmBackendModule;
    } else if (typeof WasmBackendModuleThreadedSimd !== "undefined") {
      actualModule = WasmBackendModuleThreadedSimd;
    } else {
      throw new Error("Could not find wasm module in post.js");
    }
    if (listenersAdded) {
      var tmpDispose = actualModule["_dispose"];
      actualModule["_dispose"] = function () {
        tmpDispose();
        listenersAdded.uncaughtException.forEach(function (listener) {
          process.removeListener("uncaughtException", listener);
        });
        listenersAdded.unhandledRejection.forEach(function (listener) {
          process.removeListener("unhandledRejection", listener);
        });
      };
    }

    return WasmBackendModule.ready;
  };
})();
if (typeof exports === "object" && typeof module === "object")
  module.exports = WasmBackendModule;
else if (typeof define === "function" && define["amd"])
  define([], function () {
    return WasmBackendModule;
  });
else if (typeof exports === "object")
  exports["WasmBackendModule"] = WasmBackendModule;
