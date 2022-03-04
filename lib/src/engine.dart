// ignore_for_file: unnecessary_this

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// import {BackendTimingInfo, DataMover, KernelBackend} from './backends/backend';
// import {Environment, setEnvironmentGlobal} from './environment';
// import {getGlobalNamespace} from './global_util';
// import {Add, Cast, Identity} from './kernel_names';
// import {getGradient, getKernel, getKernelsForBackend, GradFunc, NamedAttrMap, TensorInfo} from './kernel_registry';
// import {KernelProfile, Profiler} from './profiler';
// import {backpropagateGradients, getFilteredNodesXToY, TapeNode} from './tape';
// import {DataId, setTensorTracker, Tensor, TensorTracker, Variable} from './tensor';
// import {GradSaveFunc, NamedTensorMap, NamedVariableMap, TensorContainer} from './tensor_types';
// import {getTensorsInContainer} from './tensor_util';
// import {BackendValues, DataType, DataValues} from './types';
// import * as util from './util';
// import {bytesFromStringArray, makeOnesTypedArray, now, sizeFromShape} from './util';
// import * as log from './log';

import 'dart:async';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/src/base.dart'
    show Add, Cast, Identity, setUpOpHandler;
import 'package:tensorflow_wasm/src/environment.dart';
import 'package:tensorflow_wasm/src/flags.dart';
import 'package:tensorflow_wasm/src/profile.dart';
import 'package:tensorflow_wasm/src/tape.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/tensor_util.dart';

import 'backend.dart';
import 'global_util.dart';
import 'io/io.dart';
import 'kernel_registry.dart';
import 'util_base.dart' as util;
import 'util_base.dart' show log;
import 'dart:math' as math;

/**
 * A function that computes an output. The save function is for saving tensors
 * computed in the forward pass, that we need in the backward pass.
 */
typedef ForwardFunc<T> = T Function(KernelBackend backend, GradSaveFunc save);

/**
 * @docalias (a: Tensor, b: Tensor,..., save?: Function) => {
 *   value: Tensor,
 *   gradFunc: (dy: Tensor, saved?: NamedTensorMap) => Tensor | Tensor[]
 * }
 */
typedef CustomGradientFunc<T extends Tensor> = Gradient<T> Function(
  List<Tensor> inputs,
  GradSaveFunc save,
); // inputs: Tensor|GradSaveFunc

class Gradient<T extends Tensor> {
  final T value;

  final Tensors Function(T dy, List<Tensor> saved) gradFunc;

  Gradient(this.value, this.gradFunc);
}

class WithGradients<T> {
  final T value;
  final List<Tensor> grads;

  WithGradients(this.value, this.grads);
}

class MemoryInfoEngine implements MemoryInfo {
  int numTensors;
  int numDataBuffers;
  int numBytes;
  bool unreliable;
  List<String>? reasons;

  MemoryInfoEngine({
    required this.numTensors,
    required this.numDataBuffers,
    required this.numBytes,
    required this.unreliable,
    this.reasons,
  });
}

class KernelInfo {
  final String name;
  final int bytesAdded;
  final int totalBytesSnapshot;
  final int tensorsAdded;
  final int totalTensorsSnapshot;
  final List<List<int>> inputShapes;
  final List<List<int>> outputShapes;
  Object kernelTimeMs; // int | {error: String} | Promise<int|{error: String}>
  FutureOr<String> extraInfo;

  KernelInfo({
    required this.name,
    required this.bytesAdded,
    required this.totalBytesSnapshot,
    required this.tensorsAdded,
    required this.totalTensorsSnapshot,
    required this.inputShapes,
    required this.outputShapes,
    required this.kernelTimeMs,
    required this.extraInfo,
  });
}

class ProfileInfo {
  int newBytes;
  int newTensors;
  int peakBytes;
  List<KernelInfo> kernels;
  TensorContainer? result;

  Set<String> get kernelNames {
    return this.kernels.map((k) => k.name).toSet();
  }

  ProfileInfo({
    required this.newBytes,
    required this.newTensors,
    required this.peakBytes,
    required this.kernels,
    required this.result,
  });
}

class TimingInfo implements BackendTimingInfo {
  final int wallMs;
  final Object kernelMs;
  final String Function()? getExtraProfileInfo;

  TimingInfo({
    required this.wallMs,
    required this.kernelMs,
    this.getExtraProfileInfo,
  });
}

/** @docalias Function */
typedef ScopeFn<T extends TensorContainer> = T Function();

abstract class NamedTensor {
  String get name;
  Tensor get tensor;
}

class ScopeState {
  final List<Tensor> track;
  final int id;
  String name;
  ScopeState({
    required this.track,
    required this.name,
    required this.id,
  });
}

class RegisteredKernelInvocation<I extends NamedTensorMap>
    implements KernelInvocation<I> {
  final String kernelName;
  final I inputs;
  final NamedAttrMap? attrs;

  RegisteredKernelInvocation({
    required this.kernelName,
    required this.inputs,
    this.attrs,
  });
}

class CustomGradKernelInvocation<T extends Tensors, I extends NamedTensorMap>
    implements KernelInvocation<I> {
  final String? kernelName;
  final ForwardFunc<T> forwardFunc;
  final Map<String, Tensor Function()> Function(T dy, List<Tensor> saved)
      backwardsFunc;
  final I inputs;
  final NamedAttrMap? attrs;

  CustomGradKernelInvocation({
    this.kernelName,
    required this.forwardFunc,
    required this.backwardsFunc,
    required this.inputs,
    this.attrs,
  });
}

abstract class KernelInvocation<I extends NamedTensorMap> {
  String? get kernelName;
  I get inputs;
  NamedAttrMap? get attrs;
}

// kernelInvocation is RegisteredKernelInvocation<I>
// bool isRegisteredKernelInvocation<I extends NamedTensorMap>(
//     KernelInvocation<I> kernelInvocation) {
//   return kernelInvocation.kernelName != null;
// }

class TensorInfoWithBackend with TensorInfos implements TensorInfo {
  KernelBackend backend;
  final DataId dataId;
  final List<int> shape;
  final DataType dtype;
  int bytes;

  TensorInfoWithBackend({
    required this.backend,
    required this.dataId,
    required this.shape,
    required this.dtype,
    required this.bytes,
  });
}

class EngineState {
  // Public since optimizers will use it.
  NamedVariableMap registeredVariables = {};

  int nextTapeNodeId = 0;
  int numBytes = 0;
  int numTensors = 0;
  int numStringTensors = 0;
  int numDataBuffers = 0;

  List<TapeNode>? activeTape;
  // Number of nested tf.grad() statements when computing higher-order
  // gradients. E.g. `1` for first-order gradients and `2` for second-order
  // gradients. Used to track if the tape should be removed after a backprop.
  int gradientDepth = 0;
  // Number of nested kernel calls. When kernel depth is greater than 1, we turn
  // off the tape.
  int kernelDepth = 0;

  // Keep Tensors that parallel the tapes.
  ScopeState? activeScope;
  List<ScopeState> scopeStack = [];
  /**
   * Keeps track of the number of data moves during a kernel execution. We
   * maintain a stack since kernels can call other kernels, recursively.
   */
  List<int> numDataMovesStack = [];
  int nextScopeId = 0;

  // TODO: WeakMap
  final Map<DataId, TensorInfoWithBackend> tensorInfo = {};

  bool profiling = false;
  final activeProfile = ProfileInfo(
    newBytes: 0,
    newTensors: 0,
    peakBytes: 0,
    kernels: [],
    result: null,
  );

  dispose() {
    for (final variable in this.registeredVariables.values) {
      variable.dispose();
    }
  }
}

class RegistryFactory {
  final FutureOr<KernelBackend> Function() factoryFn;
  final int priority;

  RegistryFactory(this.factoryFn, this.priority);
}

class _BackendInit {
  final String name;
  final bool asyncInit;
  _BackendInit({
    required this.name,
    required this.asyncInit,
  });
}

class Engine implements TensorTracker, DataMover {
  EngineState state;
  String? backendName;
  Map<String, KernelBackend> registry = {};
  Map<String, RegistryFactory> registryFactory = {};

  late Profiler profiler;
  KernelBackend? _backendInstance;
  Future<bool>? _pendingBackendInit;
  int _pendingBackendInitId = 0;
  final Environment ENV;

  Engine(this.ENV) : state = EngineState();

  Future<void> ready() async {
    if (this._pendingBackendInit != null) {
      return this._pendingBackendInit!.then((_) {});
    }
    if (this._backendInstance != null) {
      return;
    }
    final sortedBackends = this._getSortedBackends();

    for (int i = 0; i < sortedBackends.length; i++) {
      final backendName = sortedBackends[i];
      final success = await this._initializeBackend(backendName).success;
      if (success) {
        await this.setBackend(backendName);
        return;
      }
    }

    throw Exception(
        'Could not initialize any backends, all backend initializations ' +
            'failed.');
  }

  KernelBackend get backend {
    if (this._pendingBackendInit != null) {
      throw Exception(
          "Backend '${this.backendName}' has not yet been initialized. Make " +
              'sure to await tf.ready() or await tf.setBackend() before calling ' +
              'other methods');
    }
    if (this._backendInstance == null) {
      final _backEnd = this._initializeBackendsAndReturnBest();
      if (_backEnd.asyncInit) {
        throw Exception(
            "The highest priority backend '${_backEnd.name}' has not yet been " +
                'initialized. Make sure to await tf.ready() or ' +
                'await tf.setBackend() before calling other methods');
      }
      this.setBackend(_backEnd.name);
    }
    return this._backendInstance!;
  }

  List<String> backendNames() {
    return this.registryFactory.keys.toList();
  }

  KernelBackend? findBackend(String backendName) {
    if (!this.registry.containsKey(backendName)) {
      // If the backend hasn't been initialized but we have a registry entry for
      // it, initialize it and return it.
      if (this.registryFactory.containsKey(backendName)) {
        final asyncInit = this._initializeBackend(backendName).asyncInit;
        if (asyncInit) {
          // Backend is not ready yet.
          return null;
        }
      } else {
        return null;
      }
    }
    return this.registry[backendName];
  }

  FutureOr<KernelBackend> Function()? findBackendFactory(String backendName) {
    if (!this.registryFactory.containsKey(backendName)) {
      return null;
    }
    return this.registryFactory[backendName]?.factoryFn;
  }

  bool registerBackend(
    String backendName,
    FutureOr<KernelBackend> Function() factory, [
    int priority = 1,
  ]) {
    if (this.registryFactory.containsKey(backendName)) {
      log.warning('${backendName} backend was already registered. ' +
          'Reusing existing backend factory.');
      return false;
    }
    this.registryFactory[backendName] = RegistryFactory(factory, priority);
    return true;
  }

  Future<bool> setBackend(String backendName) async {
    if (this.registryFactory[backendName] == null) {
      throw Exception("Backend name '${backendName}' not found in registry");
    }
    this.backendName = backendName;
    if (this.registry[backendName] == null) {
      this._backendInstance = null;
      final success = this._initializeBackend(backendName).success;
      final result = success is Future<bool> ? await success : success;
      if (!result) {
        return false;
      }
    }
    this._backendInstance = this.registry[backendName];
    this._setupRegisteredKernels();
    // Reset the profiler.
    this.profiler = Profiler(this._backendInstance!);

    return true;
  }

  void _setupRegisteredKernels() {
    final kernels = getKernelsForBackend(this.backendName!);
    kernels.forEach((kernel) {
      if (kernel.setupFunc != null) {
        kernel.setupFunc!(this._backendInstance!);
      }
    });
  }

  void _disposeRegisteredKernels(String backendName) {
    final kernels = getKernelsForBackend(backendName);
    kernels.forEach((kernel) {
      if (kernel.disposeFunc != null) {
        kernel.disposeFunc!(this.registry[backendName]!);
      }
    });
  }

  /**
   * Initializes a backend by looking up the backend name in the factory
   * registry and calling the factory method. Returns a boolean representing
   * whether the initialization of the backend suceeded. Throws an error if
   * there is no backend in the factory registry.
   */

  InitResult _initializeBackend(String backendName) {
    final registryFactoryEntry = this.registryFactory[backendName];
    if (registryFactoryEntry == null) {
      throw Exception(
          'Cannot initialize backend ${backendName}, no registration found.');
    }

    try {
      final backend = registryFactoryEntry.factoryFn();
      /* Test if the factory returns a promise.
      Done in a more liberal way than
      previous 'Promise.resolve(backend)===backend'
      as we needed to account for custom Promise
      implementations (e.g. Angular) */
      if (backend is Future<KernelBackend>) {
        final promiseId = ++this._pendingBackendInitId;
        final success = backend.then((backendInstance) {
          // Outdated promise. Another backend was set in the meantime.
          if (promiseId < this._pendingBackendInitId) {
            return false;
          }
          this.registry[backendName] = backendInstance;
          this._pendingBackendInit = null;
          return true;
        }).catchError((err) {
          // Outdated promise. Another backend was set in the meantime.
          if (promiseId < this._pendingBackendInitId) {
            return false;
          }
          this._pendingBackendInit = null;
          log.warning('Initialization of backend ${backendName} failed');
          log.warning(err.stack ?? err.message);
          return false;
        });
        this._pendingBackendInit = success;
        return InitResult(success: success, asyncInit: true);
      } else {
        this.registry[backendName] = backend;
        return InitResult(success: true, asyncInit: false);
      }
    } catch (err, stackTrace) {
      log.warning(
          'Initialization of backend ${backendName} failed', err, stackTrace);
      return InitResult(success: false, asyncInit: false);
    }
  }

  void removeBackend(String backendName) {
    if (!this.registryFactory.containsKey(backendName)) {
      throw Exception('${backendName} backend not found in registry');
    }
    if (this.backendName == backendName && this._pendingBackendInit != null) {
      // There is a pending promise of the backend we want to remove. Make it
      // obsolete.
      this._pendingBackendInitId++;
    }

    if (this.registry.containsKey(backendName)) {
      this._disposeRegisteredKernels(backendName);
      this.registry[backendName]!.dispose();
      this.registry.remove(backendName);
    }

    this.registryFactory.remove(backendName);

    // Unset the backend if it is active.
    if (this.backendName == backendName) {
      this._pendingBackendInit = null;
      this.backendName = null;
      this._backendInstance = null;
    }
  }

  List<String> _getSortedBackends() {
    if (this.registryFactory.length == 0) {
      throw Exception('No backend found in registry.');
    }
    return this.registryFactory.keys.toList()
      ..sort((a, b) {
        // Highest priority comes first.
        return this.registryFactory[b]!.priority -
            this.registryFactory[a]!.priority;
      });
  }

  _BackendInit _initializeBackendsAndReturnBest() {
    final sortedBackends = this._getSortedBackends();

    for (int i = 0; i < sortedBackends.length; i++) {
      final backendName = sortedBackends[i];
      final result = this._initializeBackend(backendName);
      if (result.asyncInit || result.success == true) {
        return _BackendInit(name: backendName, asyncInit: result.asyncInit);
      }
    }
    throw Exception(
        'Could not initialize any backends, all backend initializations ' +
            'failed.');
  }

  void moveData(KernelBackend backend, DataId dataId) {
    final info = this.state.tensorInfo[dataId]!;
    final srcBackend = info.backend;
    final values = this.readSync(dataId);
    final refCount = srcBackend.refCount(dataId);
    // Delete the tensor from the old backend and move it to the new
    // backend.
    srcBackend.disposeData(dataId, force: true);
    info.backend = backend;
    backend.move(dataId, values, info.shape, info.dtype, refCount);
    if (this._shouldCheckForMemLeaks()) {
      // Track the number of moves during a kernel execution to correctly
      // detect memory leaks.
      this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
    }
  }

  // // string|ScopeFn<T>
  // Object nameOrFn , ScopeFn<T>? fn,)
  T tidy<T extends TensorContainer>(
    ScopeFn<T> fn,
    // string|ScopeFn<T>
    {
    String? name,
  }) {
    // if (fn == null) {
    //   // Called with only 1 argument.
    //   if (nameOrFn is! ScopeFn<T>) {
    //     throw Exception('Please provide a function to tidy()');
    //   }
    //   fn = nameOrFn;
    // } else {
    //   // Called with 2 arguments.
    //   if (nameOrFn is! String) {
    //     throw Exception(
    //         'When calling with two arguments, the first argument ' +
    //         'to tidy() must be a string');
    //   }
    //   if (fn is! Function) {
    //     throw Exception(
    //         'When calling with two arguments, the 2nd argument ' +
    //         'to tidy() must be a function');
    //   }
    //   name = nameOrFn as String;
    //   // TODO(nsthorat,smilkov): Do operation logging and performance
    //   // profiling.
    // }
    T? result;
    return this._scopedRun(
      () => this.startScope(name),
      () => this.endScope(result),
      () {
        result = fn();
        if (result is Future) {
          util.log.severe('Cannot return a Promise inside of tidy.');
        }
        return result as T;
      },
    );
  }

  T _scopedRun<T>(void Function() start, void Function() end, T Function() f) {
    start();
    try {
      final res = f();
      end();
      return res;
    } catch (ex) {
      end();
      rethrow;
    }
  }

  static int __nextTensorId = 0;
  int _nextTensorId() {
    return Engine.__nextTensorId++;
  }

  static int __nextVariableId = 0;
  int _nextVariableId() {
    return Engine.__nextVariableId++;
  }

  /**
   * This method is called instead of the public-facing tensor.clone() when
   * saving a tensor for backwards pass. It makes sure to add the clone
   * operation to the tape regardless of being called inside a kernel
   * execution.
   */
  Tensor _clone(Tensor x) {
    final y = ENGINE.runKernel(Identity, {'x': x}) as Tensor;
    final inputs = {'x': x};
    grad(Tensors dy, _, __) {
      return {
        'x': () {
          final dtype = 'float32';
          final gradInputs = {'x': dy};
          final attrs = {'dtype': dtype};

          return ENGINE.runKernel(
            Cast, gradInputs as NamedTensorMap,
            // tslint:disable-next-line: no-unnecessary-type-assertion
            attrs,
          ) as Tensor;
        }
      };
    }

    final saved = <Tensor>[];
    this._addTapeNode(
        this.state.activeScope!.name, inputs, [y], grad, saved, {});
    return y;
  }

  /**
   * Execute a kernel with the given name and return the output tensor.
   *
   * @param kernelName The name of the kernel to execute.
   * @param inputs A map of input names to tensors.
   * @param attrs A map of attribute names to their values. An attribute is a
   *     primitive (non-tensor) input to the kernel.
   * @param inputsToSave A list of tensors, inputs to save for the backprop
   *     computation.
   * @param outputsToSave A list of booleans, specifying which output to save
   *     for the backprop computation. These are booleans since the output
   * tensors are not visible to the user.
   */
  Tensors runKernel(
    String kernelName,
    NamedTensorMap inputs, [
    NamedAttrMap? attrs,
  ]) {
    if (this.backendName == null) {
      // backend has not been initialized yet (backend initialization is lazy
      // can be deferred until an op/ kernel is run).
      // The below getter has side effects that will try to initialize the
      // backend and set properties like this.backendName
      // tslint:disable-next-line: no-unused-expression
      this.backend;
    }
    final hasKernel = getKernel(kernelName, this.backendName!) != null;
    if (!hasKernel) {
      throw Exception(
          "Kernel '${kernelName}' not registered for backend '${this.backendName}'");
    }
    return this._runKernelFunc(RegisteredKernelInvocation(
        kernelName: kernelName, inputs: inputs, attrs: attrs));
  }

  bool _shouldCheckForMemLeaks() {
    return this.ENV.getBool('IS_TEST');
  }

  void _checkKernelForMemLeak(
    String kernelName,
    int numDataIdsBefore,
    List<TensorInfo> outInfos,
  ) {
    final numDataIdsAfter = this.backend.numDataIds();

    // Count the number of data ids associated with the result of the kernel.
    int numOutputDataIds = 0;
    outInfos.forEach((info) {
      // Complex numbers allocate 3 data ids, one for 'real', one for
      // 'imaginary', and one for the container that holds the former two.
      numOutputDataIds += (info.dtype == 'complex64' ? 3 : 1);
    });

    // Account for the number of moves during kernel execution. A "data move"
    // can happen in the middle of a kernel execution, placing a new (key,value)
    // pair in the data storage. Since data moves have net zero effect (we
    // always remove the data from the old backend), we have to cancel them out
    // when detecting memory leaks.
    final numMoves =
        this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
    final dataIdsLeaked =
        numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
    if (dataIdsLeaked > 0) {
      throw Exception(
          "Backend '${this.backendName}' has an internal memory leak " +
              "(${dataIdsLeaked} data ids) after running '${kernelName}'");
    }
  }

  /**
   * Internal helper method to execute a kernel Func
   *
   * Use `runKernel` to execute kernels from outside of engine.
   */
  T _runKernelFunc<T extends Tensors, I extends NamedTensorMap>(
    KernelInvocation<I> kernelParams,
  ) {
    late List<Tensor> outputs;
    List<Tensor> saved = [];
    final isTapeOn = this.isTapeOn();

    final startingBytecount = this.state.numBytes;
    final startingNumTensors = this.state.numTensors;

    if (this._shouldCheckForMemLeaks()) {
      this.state.numDataMovesStack.add(0);
    }

    final List<Tensor> Function() kernelFunc;
    if (this.backendName == null) {
      // backend has not been initialized yet (backend initialization is lazy
      // can be deferred until an op/ kernel is run).
      // The below getter has side effects that will try to initialize the
      // backend and set properties like this.backendName
      // tslint:disable-next-line: no-unused-expression
      this.backend;
    }

    late TensorInfos out;

    final kernelOrScopeName = kernelParams is RegisteredKernelInvocation
        ? kernelParams.kernelName!
        : (this.state.activeScope?.name ?? '');

    // Create the kernelFunc from either a registered kernel OR passed in
    // forward/backward functions (used by custom grad). In this context a
    // kernelFunc wraps a kernel implementation with some bookkeeping.

    if (kernelParams is RegisteredKernelInvocation) {
      final kernelName = kernelParams.kernelName!;
      final inputs = kernelParams.inputs;

      if (this.backendName == null) {
        // backend has not been initialized yet (backend initialization is lazy
        // can be deferred until an op/ kernel is run).
        // The below getter has side effects that will try to initialize the
        // backend and set properties like this.backendName
        // tslint:disable-next-line: no-unused-expression
        this.backend;
      }
      final kernel = getKernel(kernelName, this.backendName!)!;
      util.assert_(
          kernel != null,
          () =>
              "Cannot find registered kernel '${kernelName}' for backend '${this.backendName}'");

      kernelFunc = () {
        final numDataIdsBefore = this.backend.numDataIds();
        out = kernel.kernelFunc(
            inputs: inputs, attrs: kernelParams.attrs, backend: this.backend);
        final outInfos = out.toTensorInfoList();
        if (this._shouldCheckForMemLeaks()) {
          this._checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos);
        }

        final outTensors = outInfos.map((TensorInfo outInfo) {
          // todo (yassogba) remove this option (Tensor) when node backend
          // methods have been modularized and they all return tensorInfo.
          // TensorInfos do not have a rank attribute.
          if (outInfo is Tensor && outInfo.rank != null) {
            return outInfo as Tensor;
          }
          return this.makeTensorFromDataId(
              outInfo.dataId, outInfo.shape, outInfo.dtype);
        }).toList();

        // Save any required inputs and outputs.

        // Do not save unless we are recording to the tape. Otherwise it would
        // cause a mem leak since there would be no backprop for these tensors
        // (which would otherwise dispose them).
        if (isTapeOn) {
          final tensorsToSave =
              this._getTensorsForGradient(kernelName, inputs, outTensors);
          saved = this._saveTensorsForBackwardMode(tensorsToSave);
        }
        return outTensors;
      };
    } else {
      final forwardFunc =
          (kernelParams as CustomGradKernelInvocation).forwardFunc;
      // Running a customGrad op.
      // ignore: prefer_function_declarations_over_variables
      final GradSaveFunc saveFunc = (tensors) {
        // Do not save unless we are recording to the tape. Otherwise it would
        // cause a mem leak since we would never run backprop, which disposes
        // the kept tensors.
        if (!isTapeOn) {
          return;
        }
        saved = tensors
            .map(
              (tensor) => this.keep(this._clone(tensor)),
            )
            .toList();
      };

      kernelFunc = () {
        final numDataIdsBefore = this.backend.numDataIds();
        out = this.tidy(() => forwardFunc(this.backend, saveFunc));
        final outs = out.toTensorInfoList();
        if (this._shouldCheckForMemLeaks()) {
          // Scope name is used to print a more helpful error message if needed.
          this._checkKernelForMemLeak(
              kernelOrScopeName, numDataIdsBefore, outs);
        }
        return outs.cast();
      };
    }

    //
    // Run the kernelFunc. Optionally profiling it.
    //
    final inputs = kernelParams.inputs;
    final attrs = kernelParams.attrs;

    final backwardsFunc = kernelParams is CustomGradKernelInvocation<Tensors, I>
        ? kernelParams.backwardsFunc
        : null;

    late KernelProfile kernelProfile;
    this._scopedRun(
      // Stop recording to a tape when running a kernel.
      () => this.state.kernelDepth++,
      () => this.state.kernelDepth--,
      () {
        if (!this.ENV.getBool('DEBUG') && !this.state.profiling) {
          outputs = kernelFunc();
        } else {
          kernelProfile = this
              .profiler
              .profileKernel(kernelOrScopeName, inputs, () => kernelFunc());
          if (this.ENV.getBool('DEBUG')) {
            this.profiler.logKernelProfile(kernelProfile);
          }
          outputs = kernelProfile.outputs;
        }
      },
    );

    if (isTapeOn) {
      this._addTapeNode(
        kernelOrScopeName,
        inputs,
        outputs,
        backwardsFunc == null ? null : (t, saved, _) => backwardsFunc(t, saved),
        saved,
        attrs ?? {}, // TODO: didn't have "?? {}"
      );
    }

    if (this.state.profiling) {
      this.state.activeProfile.kernels.add(KernelInfo(
            name: kernelOrScopeName,
            bytesAdded: this.state.numBytes - startingBytecount,
            totalBytesSnapshot: this.state.numBytes,
            tensorsAdded: this.state.numTensors - startingNumTensors,
            totalTensorsSnapshot: this.state.numTensors,
            inputShapes: inputs.values.map((value) => value.shape).toList(),
            outputShapes: outputs.map((item) => item.shape).toList(),
            kernelTimeMs: kernelProfile.timeMs,
            extraInfo: kernelProfile.extraInfo,
          ));
    }
    return (out is TensorInfo ? outputs[0] : TensorList(outputs)) as T;
  }

  /**
   * Saves tensors used in forward mode for use in backward mode.
   *
   * @param tensors the list of tensors to save.
   */
  List<Tensor> _saveTensorsForBackwardMode(List<Tensor> tensors) {
    final saved = tensors
        .map(
          (tensor) => this.keep(this._clone(tensor)),
        )
        .toList();
    return saved;
  }

  /**
   * Returns a list of tensors to save for a given gradient calculation.
   *
   * @param kernelName name of kernel to look up gradient for.
   * @param inputs a map of input tensors.
   * @param outputs an array of output tensors from forward mode of kernel.
   */
  List<Tensor> _getTensorsForGradient(
    String kernelName,
    NamedTensorMap inputs,
    List<Tensor> outputs,
  ) {
    final gradConfig = getGradient(kernelName);
    if (gradConfig != null) {
      final List<String> inputsToSave = gradConfig.inputsToSave ?? [];
      final List<bool> outputsToSave = gradConfig.outputsToSave ?? [];

      // If saveAllInputs is true, all inputs will be saved. Otherwise, inputs
      // specified in inputsToSave will be saved.
      final Iterable<Tensor> inputTensorsToSave;
      if (gradConfig.saveAllInputs == true) {
        util.assert_(inputs is List,
            () => 'saveAllInputs is true, expected inputs to be an array.');

        inputTensorsToSave = inputs.values;
      } else {
        inputTensorsToSave =
            inputsToSave.map((inputName) => inputs[inputName]!);
      }

      int i = 0;
      final Iterable<Tensor> outputTensorsToSave =
          outputs.where((_) => outputsToSave[i++]);

      return [...inputTensorsToSave, ...outputTensorsToSave];
    }
    // We return an empty list rather than throw an error because the kernel we
    // are looking up may not actually be relevant to backproping through the
    // overall function
    //
    // See 'does not error if irrelevant (pruned) ops are missing grads' test
    // in gradients_test.ts for an example.
    return [];
  }

  /**
   * Internal method used by public APIs for tensor creation. Makes a new
   * tensor with the provided shape, dtype and values. It always
   * creates a new data id and writes the values to the underlying backend.
   */
  Tensor makeTensor(
    DataValues values,
    List<int> shape,
    DataType? dtype, [
    KernelBackend? backend,
  ]) {
    if (values == null) {
      throw Exception('Values passed to engine.makeTensor() are null');
    }
    dtype = dtype ?? 'float32';
    backend = backend ?? this.backend;
    var backendVals = values as BackendValues;
    if (dtype == 'string' && values[0] is String) {
      backendVals =
          (values as List<String>).map((d) => util.encodeString(d)).toList();
    }
    final dataId = backend.write(backendVals, shape, dtype);
    final t = Tensor(shape, dtype, dataId, this._nextTensorId());
    this.trackTensor(t, backend);

    // Count bytes for string tensors.
    if (dtype == 'string') {
      final info = this.state.tensorInfo[dataId]!;
      final newBytes =
          util.bytesFromStringArray(backendVals as List<Uint8List>);
      this.state.numBytes += newBytes - info.bytes;
      info.bytes = newBytes;
    }
    return t;
  }

  /**
   * Internal method used by backends. Makes a new tensor
   * that is a wrapper around an existing data id. It doesn't create
   * a new data id, only increments the ref count used in memory tracking.
   */
  Tensor makeTensorFromDataId(
    DataId dataId,
    List<int> shape,
    DataType dtype, [
    KernelBackend? backend,
  ]) {
    dtype = dtype ?? 'float32';
    final t = Tensor(shape, dtype, dataId, this._nextTensorId());
    this.trackTensor(t, backend);
    return t;
  }

  Variable makeVariable(
    Tensor initialValue, {
    bool trainable = true,
    String? name,
    DataType? dtype,
  }) {
    name = name ?? this._nextVariableId().toString();
    if (dtype != null && dtype != initialValue.dtype) {
      initialValue = initialValue.cast(dtype);
    }
    final v = Variable(initialValue, trainable, name, this._nextTensorId());
    if (this.state.registeredVariables[v.name] != null) {
      throw Exception('Variable with name ${v.name} was already registered');
    }
    this.state.registeredVariables[v.name] = v;
    this.incRef(v, this.backend);
    return v;
  }

  void trackTensor(Tensor a, KernelBackend? backend) {
    this.state.numTensors++;
    if (a.dtype == 'string') {
      this.state.numStringTensors++;
    }
    // Bytes for complex numbers are counted by their components. Bytes for
    // string tensors are counted when writing values.
    int bytes = 0;
    if (a.dtype != 'complex64' && a.dtype != 'string') {
      bytes = a.size * util.bytesPerElement(a.dtype);
    }
    this.state.numBytes += bytes;

    if (!this.state.tensorInfo.containsKey(a.dataId)) {
      this.state.numDataBuffers++;
      this.state.tensorInfo[a.dataId] = TensorInfoWithBackend(
        backend: backend ?? this.backend,
        dtype: a.dtype,
        shape: a.shape,
        bytes: bytes,
        dataId: a.dataId,
      );
    }

    if (a is! Variable) {
      this._track(a);
    }
  }

  // Track the tensor by dataId and increase the refCount for the dataId in the
  // backend.
  // TODO(pyu10055): This is currently used by makeVariable method, to increase
  // refCount on the backend for the dataId. It can potentially be replaced with
  // Identity op indead of calling backend directly.
  void incRef(Tensor a, KernelBackend? backend) {
    this.trackTensor(a, backend);
    this.backend.incRef(a.dataId);
  }

  removeDataId(DataId dataId, KernelBackend backend) {
    if (this.state.tensorInfo[dataId]?.backend == backend) {
      this.state.tensorInfo.remove(dataId);
      this.state.numDataBuffers--;
    }
  }

  void disposeTensor(Tensor a) {
    final info = this.state.tensorInfo[a.dataId];
    if (info == null) {
      return;
    }

    this.state.numTensors--;
    if (a.dtype == 'string') {
      this.state.numStringTensors--;
      this.state.numBytes -= info.bytes;
    }
    // Don't count bytes for complex numbers as they are counted by their
    // components.
    if (a.dtype != 'complex64' && a.dtype != 'string') {
      final bytes = a.size * util.bytesPerElement(a.dtype);
      this.state.numBytes -= bytes;
    }

    // Remove the reference to dataId if backend dispose the data successfully
    if (info.backend.disposeData(a.dataId)) {
      this.removeDataId(a.dataId, info.backend);
    }

    // TODO(nsthorat): Construct an error and save the stack trace for
    // debugging when in debug mode. Creating a stack trace is too expensive
    // to do unconditionally.
  }

  void disposeVariables() {
    for (final v in this.state.registeredVariables.values) {
      this.disposeVariable(v);
    }
  }

  void disposeVariable(Variable v) {
    this.disposeTensor(v);
    if (this.state.registeredVariables[v.name] != null) {
      this.state.registeredVariables.remove(v.name);
    }
  }

  MemoryInfoEngine memory() {
    final _info = this.backend.memory();
    final info = MemoryInfoEngine(
      numTensors: this.state.numTensors,
      numDataBuffers: this.state.numDataBuffers,
      numBytes: this.state.numBytes,
      unreliable: _info.unreliable,
      reasons: _info.reasons,
    );
    if (this.state.numStringTensors > 0) {
      info.unreliable = true;

      info.reasons ??= [];
      info.reasons!.add('Memory usage by string tensors is approximate ' +
          '(2 bytes per character)');
    }
    return info;
  }

  Future<ProfileInfo> profile(
    FutureOr<TensorContainer> Function() query,
  ) async {
    this.state.profiling = true;

    final startBytes = this.state.numBytes;
    final startNumTensors = this.state.numTensors;
    final activeProfile = this.state.activeProfile;
    activeProfile.kernels = [];
    activeProfile.result = await query();

    this.state.profiling = false;

    activeProfile.peakBytes =
        activeProfile.kernels.map((d) => d.totalBytesSnapshot).reduce(math.max);
    activeProfile.newBytes = this.state.numBytes - startBytes;
    activeProfile.newTensors = this.state.numTensors - startNumTensors;
    for (final kernel in activeProfile.kernels) {
      kernel.kernelTimeMs = kernel.kernelTimeMs is Future
          ? await kernel.kernelTimeMs
          : kernel.kernelTimeMs;
      kernel.extraInfo = await kernel.extraInfo;
    }
    return activeProfile;
  }

  bool isTapeOn() {
    return this.state.gradientDepth > 0 && this.state.kernelDepth == 0;
  }

  void _addTapeNode(
    String kernelName,
    NamedTensorMap inputs,
    List<Tensor> outputs,
    GradFunc? gradientsFunc,
    List<Tensor> saved,
    NamedAttrMap attrs,
  ) {
    final tapeNode = TapeNode(
      id: this.state.nextTapeNodeId++,
      kernelName: kernelName,
      inputs: inputs,
      outputs: outputs,
      saved: saved,
    );

    final gradConfig = getGradient(kernelName);
    if (gradConfig != null) {
      gradientsFunc = gradConfig.gradFunc;
    }
    if (gradientsFunc != null) {
      tapeNode.gradient = (List<Tensor> dys) {
        // TODO(smilkov): To optimize back-prop, pass dys that are not used in
        // the backprop graph to the user as null instead of zeros
        dys = dys.mapIndexed((i, dy) {
          if (dy == null) {
            final output = outputs[i];
            final vals = util.makeZerosTypedArray(output.size, output.dtype);
            return this.makeTensor(vals, output.shape, output.dtype);
          }
          return dy;
        }).toList();
        // Grad functions of ops with single outputs expect a dy, while ops
        // with multiple outputs expect dys (array of dy).
        return gradientsFunc!(
          dys.length > 1 ? TensorList(dys) : dys[0],
          saved,
          attrs,
        );
      };
    }
    this.state.activeTape!.add(tapeNode);
  }

  T keep<T extends Tensor>(T result) {
    result.kept = true;
    return result;
  }

  _startTape() {
    if (this.state.gradientDepth == 0) {
      this.state.activeTape = [];
    }
    this.state.gradientDepth++;
  }

  _endTape() {
    this.state.gradientDepth--;
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope([String? name]) {
    final scopeInfo = ScopeState(
      track: [],
      name: 'unnamed scope',
      id: this.state.nextScopeId++,
    );
    if (name != null) {
      scopeInfo.name = name;
    }
    this.state.scopeStack.add(scopeInfo);
    this.state.activeScope = scopeInfo;
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope([TensorContainer? result]) {
    final tensorsToTrackInParent = getTensorsInContainer(result);
    final tensorsToTrackInParentSet =
        tensorsToTrackInParent.map((t) => t.id).toSet();

    // Dispose the arrays tracked in this scope.
    for (int i = 0; i < this.state.activeScope!.track.length; i++) {
      final tensor = this.state.activeScope!.track[i];
      if (!tensor.kept && !tensorsToTrackInParentSet.contains(tensor.id)) {
        tensor.dispose();
      }
    }

    final oldScope = this.state.scopeStack.removeLast();
    this.state.activeScope = this.state.scopeStack.length == 0
        ? null
        : this.state.scopeStack[this.state.scopeStack.length - 1];

    // Track the current result in the parent scope.
    tensorsToTrackInParent.forEach((tensor) {
      // Only track the tensor if was allocated in the inner scope and is not
      // globally kept.
      if (!tensor.kept && tensor.scopeId == oldScope.id) {
        this._track(tensor);
      }
    });
  }

  /**
   * Returns gradients of `f` with respect to each of the `xs`. The gradients
   * returned are of the same length as `xs`, but some might be null if `f`
   * was not a function of that `x`. It also takes optional dy to multiply the
   * gradient, which defaults to `1`.
   */
  WithGradients<T> gradients<T extends Tensor>(
    T Function() f,
    List<Tensor> xs, {
    T? dy,
    bool allowNoGradients = false,
  }) {
    util.assert_(
        xs.length > 0, () => 'gradients() received an empty list of xs.');
    if (dy != null && dy.dtype != 'float32') {
      throw Exception("dy must have 'float32' dtype, but has '${dy.dtype}'");
    }

    final y = this._scopedRun(
      () => this._startTape(),
      () => this._endTape(),
      () => this.tidy(f, name: 'forward'),
    );

    util.assert_(
        y is Tensor, () => 'The result y returned by f() must be a tensor.');
    // Filter out the nodes that don't connect x => y.
    final filteredTape = getFilteredNodesXToY(this.state.activeTape!, xs, y);
    if (!allowNoGradients && filteredTape.length == 0 && xs.length > 0) {
      throw Exception(
          'Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
              'that the f you passed encloses all operations that lead from x ' +
              'to y.');
    }

    return this.tidy(() {
      final accumulatedGradientMap = <int, Tensor>{};
      accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;

      // Backprop gradients through the filtered nodes.
      backpropagateGradients(
        accumulatedGradientMap,
        filteredTape,
        // Pass the tidy function to avoid circular dep with `tape.ts`.
        (f) => this.tidy(f as ScopeFn<Tensor>),
        // Pass an add function to avoide a circular dep with `tape.ts`.
        add,
      );
      final grads = xs.map((x) => accumulatedGradientMap[x.id]!).toList();

      if (this.state.gradientDepth == 0) {
        // This means that we are not computing higher-order gradients
        // and can clean up the tape.
        this.state.activeTape!.forEach((node) {
          for (final tensor in node.saved!) {
            tensor.dispose();
          }
        });
        this.state.activeTape = null;
      }
      return WithGradients(y, grads);
    }, name: 'backward');
  }

  // TODO: (...args: Array<Tensor|GradSaveFunc>) => T
  T Function(List<Tensor>) customGrad<T extends Tensor>(
    CustomGradientFunc<T> f,
  ) {
    util.assert_(f is Function,
        () => 'The f passed in customGrad(f) must be a function.');
    return (List<Tensor> inputs) {
      util.assert_(
          inputs.every((t) => t is Tensor),
          () =>
              'The args passed in customGrad(f)(x1, x2,...) must all be ' +
              'tensors');

      late Gradient<T> res;
      final NamedTensorMap inputMap = {};
      inputs.forEachIndexed((i, input) {
        inputMap[i.toString()] = input; // TODO: wasn't i.toString()
      });

      // ignore: prefer_function_declarations_over_variables
      final ForwardFunc<T> forwardFunc = (_, save) {
        res = f(inputs, save);
        util.assert_(
            res.value is Tensor,
            () =>
                'The function f passed in customGrad(f) must return an ' +
                'object where `obj.value` is a tensor');
        util.assert_(
            res.gradFunc is Function,
            () =>
                'The function f passed in customGrad(f) must return an ' +
                'object where `obj.gradFunc` is a function.');
        return res.value;
      };

      backwardsFunc(T dy, List<Tensor> saved) {
        final gradRes = res.gradFunc(dy, saved);
        final List<Tensor> grads = gradRes.match(
          (tensor) => [tensor],
          (list) => list,
        );
        util.assert_(
            grads.length == inputs.length,
            () =>
                'The function f passed in customGrad(f) must return an ' +
                'object where `obj.gradFunc` is a function that returns ' +
                'the same number of tensors as inputs passed to f(...).');
        util.assert_(
            grads.every((t) => t is Tensor),
            () =>
                'The function f passed in customGrad(f) must return an ' +
                'object where `obj.gradFunc` is a function that returns ' +
                'a list of only tensors.');
        final gradMap = <String, Tensor Function()>{};
        grads.forEachIndexed((i, grad) {
          gradMap[i.toString()] = () => grad; // TODO: wasn't i.toString()
        });
        return gradMap;
      }

      return this._runKernelFunc(CustomGradKernelInvocation(
        forwardFunc: forwardFunc,
        backwardsFunc: backwardsFunc,
        inputs: inputMap,
      ));
    };
  }

  BackendValues readSync(DataId dataId) {
    // Route the read to the correct backend.
    final info = this.state.tensorInfo[dataId]!;
    return info.backend.readSync(dataId);
  }

  Future<BackendValues> read(DataId dataId) {
    // Route the read to the correct backend.
    final info = this.state.tensorInfo[dataId]!;
    return info.backend.read(dataId);
  }

  Future<TimingInfo> time(void Function() query) async {
    final start = util.now();
    final timingInfo = await this.backend.time(query);
    final wallMs = util.now() - start;
    return TimingInfo(
      kernelMs: timingInfo.kernelMs,
      getExtraProfileInfo: timingInfo.getExtraProfileInfo,
      wallMs: wallMs,
    );
  }

  /**
   * Tracks a Tensor in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The Tensor to track in the current scope.
   */
  T _track<T extends Tensor>(T result) {
    if (this.state.activeScope != null) {
      result.scopeId = this.state.activeScope!.id;
      this.state.activeScope!.track.add(result);
    }

    return result;
  }

  NamedVariableMap get registeredVariables {
    return this.state.registeredVariables;
  }

  /**
   * Resets the engine state. Removes all backends but does not remove
   * registered backend factories.
   */
  void reset() {
    // Make any pending promise obsolete.
    this._pendingBackendInitId++;

    this.state.dispose();
    this.ENV.reset();
    this.state = EngineState();

    for (final backendName in this.registry.keys) {
      this._disposeRegisteredKernels(backendName);
      this.registry[backendName]!.dispose();
      this.registry.remove(backendName);
    }
    this.backendName = null;
    this._backendInstance = null;
    this._pendingBackendInit = null;
  }
}

class InitResult {
  final FutureOr<bool> success;
  final bool asyncInit;

  InitResult({required this.success, required this.asyncInit});
}

Tensor ones(List<int> shape) {
  final values = util.makeOnesTypedArray(util.sizeFromShape(shape), 'float32');
  return ENGINE.makeTensor(values, shape, 'float32');
}

Engine _getOrMakeEngine() {
  final ns = getGlobalNamespace();
  if (ns.tfengine == null) {
    final environment = Environment(ns);
    setEnvironmentGlobal(environment);
    ns.tfengine = Engine(environment);
    setUpOpHandler();
    setUpFlags();
    setUpIo();
  }
  final engine = ns.tfengine!;

  // Tell the current tensor interface that the global engine is responsible
  // for tracking.
  setTensorTracker(() => engine);
  return engine;
}

final ENGINE = _getOrMakeEngine();

/**
 * A implementation of the add op for use within engine and tape.
 *
 * This allows us to avoid a circular dependency between add.ts and engine.
 * It is exported to be available in tape tests.
 */
Tensor add(Tensor a, Tensor b) {
  // We duplicate Add here to avoid a circular dependency with add.ts.
  final inputs = {'a': a, 'b': b};
  return ENGINE.runKernel(Add, inputs) as Tensor;
}
