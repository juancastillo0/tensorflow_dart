import 'dart:async';
import 'dart:typed_data';

import 'package:tensorflow_wasm/src/backend.dart';
import 'package:tensorflow_wasm/src/tensor.dart';

import 'emscripten_module.dart';
import 'util_base.dart' as util;

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// import './flags_wasm';

// import {backend_util, BackendTimingInfo, DataStorage, DataType, deprecationWarn, engine, env, KernelBackend, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasmModule, WasmFactoryConfig} from '../wasm-out/tfjs-backend-wasm';
// import {BackendWasmThreadedSimdModule} from '../wasm-out/tfjs-backend-wasm-threaded-simd';
// import wasmFactoryThreadedSimd from '../wasm-out/tfjs-backend-wasm-threaded-simd.js';
// // @ts-ignore
// import {wasmWorkerContents} from '../wasm-out/tfjs-backend-wasm-threaded-simd.worker.js';
// import wasmFactory from '../wasm-out/tfjs-backend-wasm.js';

class TensorData {
  final int id;
  final int? memoryOffset;
  final List<int> shape;
  final DataType dtype;
  int refCount;
  /** Only used for string tensors, storing encoded bytes. */
  final List<Uint8List>? stringBytes;

  TensorData({
    required this.id,
    required this.memoryOffset,
    required this.shape,
    required this.dtype,
    required this.refCount,
    this.stringBytes,
  });
}

class BackendWasm extends KernelBackend {
  // TODO: extends KernelBackend
  // 0 is reserved for null data ids.
  int dataIdNextNumber = 1;
  final BackendWasmModule wasm; // |BackendWasmThreadedSimdModule
  late final DataStorage<TensorData> dataIdMap;

  BackendWasm(this.wasm) {
    // super();
    this.wasm.tfjs.initWithThreadsCount(threadsCount);
    actualThreadsCount = this.wasm.tfjs.getThreadsCount();
    this.dataIdMap = DataStorage(this, engine());
  }

  DataId write(
    BackendValues values,
    List<int> shape,
    DataType dtype,
  ) {
    final dataId = {'id': this.dataIdNextNumber++};
    this.move(dataId, values, shape, dtype, 1);
    return dataId;
  }

  int numDataIds() {
    return this.dataIdMap.numDataIds();
  }

  Future<BackendTimingInfo> time(void Function() f) async {
    final start = util.now();
    f();
    final kernelMs = util.now() - start;
    return BackendTimingInfo(kernelMs);
  }

  void move(
    DataId dataId,
    BackendValues values,
    List<int> shape,
    DataType dtype,
    int refCount,
  ) {
    final id = this.dataIdNextNumber++;
    if (dtype == 'string') {
      final stringBytes = values as List<Uint8List>;
      this.dataIdMap.set(
          dataId,
          TensorData(
              id: id,
              stringBytes: stringBytes,
              shape: shape,
              dtype: dtype,
              memoryOffset: null,
              refCount: refCount));
      return;
    }

    final size = util.sizeFromShape(shape);
    final numBytes = size * util.bytesPerElement(dtype);
    final memoryOffset = this.wasm.malloc(numBytes);

    this.dataIdMap.set(
          dataId,
          TensorData(
            id: id,
            memoryOffset: memoryOffset,
            shape: shape,
            dtype: dtype,
            refCount: refCount,
          ),
        );

    this.wasm.tfjs.registerTensor(id, size, memoryOffset);

    if (values != null) {
      this.wasm.HEAPU8.set(
            Uint8Array(
              (values as TypedData).buffer,
              (values as TypedData).offsetInBytes,
              numBytes,
            ),
            memoryOffset,
          );
    }
  }

  Future<BackendValues> read(DataId dataId) async {
    return this.readSync(dataId);
  }

  BackendValues readSync(DataId dataId, [int? start, int? end]) {
    final data = this.dataIdMap.get(dataId)!;
    final memoryOffset = data.memoryOffset!;
    final dtype = data.dtype;
    final shape = data.shape;
    final stringBytes = data.stringBytes;

    if (dtype == 'string') {
      // Slice all elements.
      if ((start == null || start == 0) &&
          (end == null || end >= stringBytes!.length)) {
        return stringBytes;
      }
      return stringBytes!.sublist(start!, end);
    }
    start = start ?? 0;
    end = end ?? util.sizeFromShape(shape);
    final bytesPerElement = util.bytesPerElement(dtype);
    final bytes = this.wasm.HEAPU8.sublist(
          memoryOffset + start * bytesPerElement,
          memoryOffset + end * bytesPerElement,
        );
    return typedArrayFromBuffer(bytes.buffer, dtype);
  }

  /**
   * Dispose the memory if the dataId has 0 refCount. Return true if the memory
   * is released, false otherwise.
   * @param dataId
   * @oaram force Optional, remove the data regardless of refCount
   */
  bool disposeData(DataId dataId, {bool force = false}) {
    if (this.dataIdMap.has(dataId)) {
      final data = this.dataIdMap.get(dataId)!;
      data.refCount--;
      if (!force && data.refCount > 0) {
        return false;
      }

      this.wasm.free(data.memoryOffset!);
      this.wasm.tfjs.disposeData(data.id);
      this.dataIdMap.delete(dataId);
    }
    return true;
  }

  /** Return refCount of a `TensorData`. */
  int refCount(DataId dataId) {
    if (this.dataIdMap.has(dataId)) {
      final tensorData = this.dataIdMap.get(dataId)!;
      return tensorData.refCount;
    }
    return 0;
  }

  void incRef(DataId dataId) {
    final data = this.dataIdMap.get(dataId);
    if (data != null) {
      data.refCount++;
    }
  }

  int floatPrecision() {
    return 32;
  }

  // Returns the memory offset of a tensor. Useful for debugging and unit
  // testing.
  int getMemoryOffset(DataId dataId) {
    return this.dataIdMap.get(dataId)!.memoryOffset!;
  }

  void dispose() {
    this.wasm.tfjs.dispose();
    // TODO:
    // if ('PThread' in this.wasm) {
    //   this.wasm.PThread.terminateAllThreads();
    // }
    // this.wasm = null;
  }

  MemoryInfo memory() {
    return MemoryInfo(unreliable: false);
  }

  /**
   * Make a tensor info for the output of an op. If `memoryOffset` is not
   * present, this method allocates memory on the WASM heap. If `memoryOffset`
   * is present, the memory was allocated elsewhere (in c++) and we just record
   * the pointer where that memory lives.
   */
  TensorInfo makeOutput(List<int> shape, DataType dtype, int? memoryOffset) {
    final Map dataId;
    if (memoryOffset == null) {
      dataId = this.write(null /* values */, shape, dtype);
    } else {
      final id = this.dataIdNextNumber++;
      dataId = {'id': id};
      this.dataIdMap.set(
            dataId,
            TensorData(
                id: id,
                memoryOffset: memoryOffset,
                shape: shape,
                dtype: dtype,
                refCount: 1),
          );
      final size = util.sizeFromShape(shape);
      this.wasm.tfjs.registerTensor(id, size, memoryOffset);
    }
    return TensorInfo(
      dataId: dataId,
      shape: shape,
      dtype: dtype,
    );
  }

  TypedData typedArrayFromHeap(TensorInfo _info) {
    final shape = _info.shape;
    final dtype = _info.dtype;
    final dataId = _info.dataId;

    final buffer = this.wasm.HEAPU8.buffer;
    final memoryOffset = this.dataIdMap.get(dataId)!.memoryOffset!;
    final size = util.sizeFromShape(shape);
    switch (dtype) {
      case 'float32':
        return Float32List.view(buffer, memoryOffset, size);
      case 'int32':
        return Int32List.view(buffer, memoryOffset, size);
      case 'bool':
        return Uint8List.view(buffer, memoryOffset, size);
      default:
        throw Exception('Unknown dtype ${dtype}');
    }
  }
}

// createInstantiateWasmFunc(String path) {
//   // this will be replace by rollup plugin patchWechatWebAssembly in
//   // minprogram's output.
//   // tslint:disable-next-line:no-any
//   return (imports: any, callback: any) {
//     util.fetch(path, {'credentials': 'same-origin'}).then((response) {
//       if (!response['ok']) {
//         imports.env.a("failed to load wasm binary file at '${path}'");
//       }
//       response.arrayBuffer().then((binary)  {
//         WebAssembly.instantiate(binary, imports).then((output) {
//           callback(output.instance, output.module);
//         });
//       });
//     });
//     return {};
//   };
// }

/**
 * Returns the path of the WASM binary.
 * @param simdSupported whether SIMD is supported
 * @param threadsSupported whether multithreading is supported
 * @param wasmModuleFolder the directory containing the WASM binaries.
 */
String getPathToWasmBinary(
  bool simdSupported,
  bool threadsSupported,
  String wasmModuleFolder,
) {
  if (wasmPath != null) {
    // If wasmPath is defined, the user has supplied a full path to
    // the vanilla .wasm binary.
    return wasmPath!;
  }

  String path = 'tfjs-backend-wasm.wasm';
  if (simdSupported && threadsSupported) {
    path = 'tfjs-backend-wasm-threaded-simd.wasm';
  } else if (simdSupported) {
    path = 'tfjs-backend-wasm-simd.wasm';
  }

  if (wasmFileMap != null) {
    if (wasmFileMap[path] != null) {
      return wasmFileMap[path]!;
    }
  }

  return wasmModuleFolder + path;
}

/**
 * Initializes the wasm module and creates the js <--> wasm bridge.
 *
 * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
 * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested
 * in Chrome 76).
 */
Future<BackendWasmModule> init() async {
  // final [simdSupported, threadsSupported] = await Future.wait([
  //   env().getAsync('WASM_HAS_SIMD_SUPPORT'),
  //   env().getAsync('WASM_HAS_MULTITHREAD_SUPPORT')
  // ]);
  // TODO:
  final simdSupported = false;
  final threadsSupported = false;

  final completer = Completer<BackendWasmModule>();

  /**
     * This function overrides the Emscripten module locateFile utility.
     * @param path The relative path to the file that needs to be loaded.
     * @param prefix The path to the main JavaScript file's directory.
     */
  String locateFile(String path, String prefix) {
    if (path.endsWith('.worker.js')) {
      final response = wasmWorkerContents;
      final blob = Blob([response], {'type': 'application/javascript'});
      return URL.createObjectURL(blob);
    }

    if (path.endsWith('.wasm')) {
      return getPathToWasmBinary(
          simdSupported as bool,
          threadsSupported as bool,
          wasmPathPrefix != null ? wasmPathPrefix! : prefix);
    }
    return prefix + path;
  }

  // Use the instantiateWasm override when system fetch is not available.
  // Reference:
  // https://github.com/emscripten-core/emscripten/blob/2bca083cbbd5a4133db61fbd74d04f7feecfa907/tests/manual_wasm_instantiate.html#L170
  // TODO:
  // if (customFetch) {
  //   factoryConfig.instantiateWasm =
  //       createInstantiateWasmFunc(getPathToWasmBinary(
  //           simdSupported as bool, threadsSupported as bool,
  //           wasmPathPrefix != null ? wasmPathPrefix : ''));
  // }

  bool initialized = false;
  onAbort(String error) {
    if (initialized) {
      // Emscripten already called console.warn so no need to double log.
      return;
    }
    if (initAborted) {
      // Emscripten calls `onAbort` twice, resulting in double error
      // messages.
      return;
    }
    initAborted = true;
    final rejectMsg =
        'Make sure the server can serve the `.wasm` file relative to the ' +
            'bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers';
    completer.completeError({'message': rejectMsg});
  }

  final factoryConfig = WasmFactoryConfig(
    onAbort: onAbort,
    locateFile: locateFile,
  );

  final Future<EmscriptenModule> wasm;
  // If `wasmPath` has been defined we must initialize the vanilla module.
  if (threadsSupported && simdSupported && wasmPath == null) {
    // factoryConfig.mainScriptUrlOrBlob = new Blob(
    //     ['var WasmBackendModuleThreadedSimd = ' +
    //      wasmFactoryThreadedSimd.toString()],
    //     {'type': 'text/javascript'});
    // wasm = wasmFactoryThreadedSimd(factoryConfig);
  } else {
    // The wasmFactory works for both vanilla and SIMD binaries.
    wasm = wasmFactory(factoryConfig);
  }

  // The WASM module has been successfully created by the factory.
  // Any error will be caught by the onAbort callback defined above.
  wasm.then((module) {
    initialized = true;
    initAborted = false;

    completer.complete(BackendWasmModule(module));
  });

  return completer.future;
}

TypedData typedArrayFromBuffer(ByteBuffer buffer, DataType dtype) {
  switch (dtype) {
    case 'float32':
      return Float32List.view(buffer);
    case 'int32':
      return Int32List.view(buffer);
    case 'bool':
      return Uint8List.view(buffer);
    default:
      throw Exception('Unknown dtype ${dtype}');
  }
}

class BackendWasmModule extends EmscriptenModule {
  BackendWasmModule(EmscriptenModule module) : super.fromModule(module);

  // Using the tfjs namespace to avoid conflict with emscripten's API.
  late final tfjs = TFModule(this);
}

class TFModule {
  final EmscriptenModule module;

  static const String? voidReturnType = null;

  TFModule(this.module);

  late final init = module.cwrap('init', null, []);
  late final initWithThreadsCount =
      module.cwrap('init_with_threads_count', null, ['number']);
  late final getThreadsCount = module.cwrap('get_threads_count', 'number', []);
  late final registerTensor = module.cwrap(
    'register_tensor',
    null,
    [
      'number', // id
      'number', // size
      'number', // memoryOffset
    ],
  );
  late final disposeData =
      module.cwrap('dispose_data', voidReturnType, ['number']);
  late final dispose = module.cwrap('dispose', voidReturnType, []);
}

const wasmBinaryNames = [
  'tfjs-backend-wasm.wasm',
  'tfjs-backend-wasm-simd.wasm',
  'tfjs-backend-wasm-threaded-simd.wasm'
];
// type WasmBinaryName = typeof wasmBinaryNames[number];

String? wasmPath = null;
String? wasmPathPrefix = null;
Map<String, String> wasmFileMap =
    {}; //: {[key in WasmBinaryName]?: string} = {};
bool initAborted = false;
bool customFetch = false;

void deprecationWarn(String warning) {
  print(warning);
}

/**
 * @deprecated Use `setWasmPaths` instead.
 * Sets the path to the `.wasm` file which will be fetched when the wasm
 * backend is initialized. See
 * https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers
 * for more details.
 * @param path wasm file path or url
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
void setWasmPath(String path, [bool usePlatformFetch = false]) {
  deprecationWarn(
      'setWasmPath has been deprecated in favor of setWasmPaths and' +
          ' will be removed in a future release.');
  if (initAborted) {
    throw Exception(
        'The WASM backend was already initialized. Make sure you call ' +
            '`setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`');
  }
  wasmPath = path;
  customFetch = usePlatformFetch;
}

/**
 * Configures the locations of the WASM binaries.
 *
 * ```js
 * setWasmPaths({
 *  'tfjs-backend-wasm.wasm': 'renamed.wasm',
 *  'tfjs-backend-wasm-simd.wasm': 'renamed-simd.wasm',
 *  'tfjs-backend-wasm-threaded-simd.wasm': 'renamed-threaded-simd.wasm'
 * });
 * tf.setBackend('wasm');
 * ```
 *
 * @param prefixOrFileMap This can be either a string or object:
 *  - (string) The path to the directory where the WASM binaries are located.
 *     Note that this prefix will be used to load each binary (vanilla,
 *     SIMD-enabled, threading-enabled, etc.).
 *  - (object) Mapping from names of WASM binaries to custom
 *     full paths specifying the locations of those binaries. This is useful if
 *     your WASM binaries are not all located in the same directory, or if your
 *     WASM binaries have been renamed.
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
void setWasmPaths(
  //string|{[key in WasmBinaryName]?: string},
  Object prefixOrFileMap, [
  bool usePlatformFetch = false,
]) {
  if (initAborted) {
    throw Exception(
        'The WASM backend was already initialized. Make sure you call ' +
            '`setWasmPaths()` before you call `tf.setBackend()` or ' +
            '`tf.ready()`');
  }

  if (prefixOrFileMap is String) {
    wasmPathPrefix = prefixOrFileMap;
  } else {
    wasmFileMap = (prefixOrFileMap as Map).cast();
    final missingPaths =
        wasmBinaryNames.where((name) => wasmFileMap[name] == null);
    if (missingPaths.length > 0) {
      throw Exception('There were no entries found for the following binaries: ' +
          '${missingPaths.join(',')}. Please either call setWasmPaths with a ' +
          'map providing a path for each binary, or with a string indicating ' +
          'the directory where all the binaries can be found.');
    }
  }

  customFetch = usePlatformFetch;
}

/** Used in unit tests. */
void resetWasmPath() {
  wasmPath = null;
  wasmPathPrefix = null;
  wasmFileMap = {};
  customFetch = false;
  initAborted = false;
}

int threadsCount = -1;
int actualThreadsCount = -1;

/**
 * Sets the number of threads that will be used by XNNPACK to create
 * threadpool (default to the number of logical CPU cores).
 *
 * This must be called before calling `tf.setBackend('wasm')`.
 */
void setThreadsCount(int numThreads) {
  threadsCount = numThreads;
}

/**
 * Gets the actual threads count that is used by XNNPACK.
 *
 * It is set after the backend is intialized.
 */
int getThreadsCount() {
  if (actualThreadsCount == -1) {
    throw Exception('WASM backend not initialized.');
  }
  return actualThreadsCount;
}
