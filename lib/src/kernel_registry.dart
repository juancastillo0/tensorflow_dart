import 'package:tensorflow_wasm/src/environment.dart';
import 'package:tensorflow_wasm/src/tape.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/src/global_util.dart';
import 'package:tensorflow_wasm/src/tensor.dart';

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
 * ==========================================================================
 */
// import {env} from './environment';

// import {getGlobal} from './global_util';
// import {NamedGradientMap} from './tape';
// import {Tensor} from './tensor';
// import {DataType, RecursiveArray} from './types';

final kernelRegistry =
    getGlobal('kernelRegistry', () => Map<String, KernelConfig>());
final gradRegistry = getGlobal('gradRegistry', () => Map<String, GradConfig>());

typedef AttributeValue = Object?;
// number|number[]|boolean|boolean[]|String|string[]|NamedAttrMap;

/** These are extra non-tensor/primitive params passed to kernel functions. */
typedef Attribute
    = AttributeValue; // AttributeValue|RecursiveArray<AttributeValue>;

/** Specifies the code to run when executing a kernel. */
typedef KernelFunc = TensorInfos Function({
  required NamedTensorInfoMap inputs,
  required Object backend,
  NamedAttrMap? attrs,
});

/** The function to run when computing a gradient during backprop. */
typedef GradFunc = NamedGradientMap Function(
  Tensors dy,
  List<Tensor> saved,
  NamedAttrMap attrs,
);

/** Function that gets called after the backend initializes. */
typedef KernelSetupFunc = void Function(Object backend);
/** Function that gets called right before the backend is disposed. */
typedef KernelDisposeFunc = KernelSetupFunc;

/** Config object for registering a kernel in the global registry. */
class KernelConfig {
  final String kernelName;
  final String backendName;
  final KernelFunc kernelFunc;
  final KernelSetupFunc? setupFunc;
  final KernelDisposeFunc? disposeFunc;

  KernelConfig({
    required this.kernelName,
    required this.backendName,
    required this.kernelFunc,
    this.setupFunc,
    this.disposeFunc,
  });
}

extension CopyWithKernelConfig on KernelConfig {
  KernelConfig withNewBackendName(String newBackendName) {
    return KernelConfig(
      kernelName: kernelName,
      backendName: newBackendName,
      kernelFunc: kernelFunc,
      setupFunc: setupFunc,
      disposeFunc: disposeFunc,
    );
  }
}

/** Config object for registering a gradient in the global registry. */
class GradConfig {
  final String kernelName;
  final List<String>? inputsToSave;
  // When saveAllInputs is true, all inputs will be saved. Only use this flag
  // if inputs is an array of Tensors.
  final bool? saveAllInputs;
  final List<bool>? outputsToSave;
  final GradFunc gradFunc;

  GradConfig({
    required this.kernelName,
    this.inputsToSave,
    this.saveAllInputs,
    this.outputsToSave,
    required this.gradFunc,
  });
}

/** Holds metadata for a given tensor. */
// export interface TensorInfo {
//   dataId: DataId;
//   shape: number[];
//   dtype: DataType;
// }

typedef NamedTensorInfoMap = Map<String, TensorInfo>;

typedef NamedAttrMap = Map<String, Attribute>;

/**
 * Returns the kernel function (code) associated with the provided names.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 */
KernelConfig? getKernel(String kernelName, String backendName) {
  final key = makeKey(kernelName, backendName);
  return kernelRegistry[key];
}

/**
 * Returns the registered gradient info associated with the provided kernel.
 * @param kernelName The official TF kernel name.
 */
GradConfig? getGradient(String kernelName) {
  return gradRegistry[kernelName];
}

List<KernelConfig> getKernelsForBackend(String backendName) {
  final result = <KernelConfig>[
    ...kernelRegistry.entries
        .where((e) => e.key.split('_').first == backendName)
        .map((e) => e.value)
  ];

  return result;
}

/**
 * Registers the function (forward pass) for the kernel in a global registry.
 *
 * @param config A config object with the following properties:
 * - `kernelName` The official name of the kernel.
 * - `backendName` The official name of the backend.
 * - `kernelFunc` The function to run during the forward pass of the kernel.
 * - `setupFunc` Optional. Gets called once, after the backend initializes.
 * - `disposeFunc` Optional. Gets called once, right before the backend is
 * disposed.
 */
void registerKernel(KernelConfig config) {
  final kernelName = config.kernelName;
  final backendName = config.backendName;

  final key = makeKey(kernelName, backendName);
  if (kernelRegistry.containsKey(key)) {
    util.log.warning("The kernel '${kernelName}' for backend " +
        "'${backendName}' is already registered");
  }
  kernelRegistry[key] = config;
}

/**
 * Registers a gradient function for a given kernel in the global registry,
 * to be used during the back-propagation of that kernel.
 *
 * @param config An object with the following properties:
 * - `kernelName` The name of the kernel that the gradient function is for.
 * - `gradFunc` The function to run during back-propagation.
 */
void registerGradient(GradConfig config) {
  final kernelName = config.kernelName;

  if (gradRegistry.containsKey(kernelName)) {
    // TODO (yassogba) after 3.0 assess whether we need to keep this gated
    // to debug mode.
    if (env().getBool('DEBUG')) {
      util.log.warning("Overriding the gradient for '${kernelName}'");
    }
  }
  gradRegistry[kernelName] = config;
}

/**
 * Removes the kernel function from the registry.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 *
 */
void unregisterKernel(String kernelName, String backendName) {
  final key = makeKey(kernelName, backendName);
  if (!kernelRegistry.containsKey(key)) {
    throw Exception("The kernel '${kernelName}' for backend " +
        "'${backendName}' is not registered");
  }
  kernelRegistry.remove(key);
}

/** Removes the registered gradient from the global registry. */
void unregisterGradient(String kernelName) {
  if (!gradRegistry.containsKey(kernelName)) {
    throw Exception(
        "The gradient '${kernelName}' for backend is not registered");
  }
  gradRegistry.remove(kernelName);
}

/**
 * Finds kernels that have already been registered to a backend and re-registers
 * them for a new backend. Useful for registering custom backends.
 * @param registeredBackendName Already registered backend.
 * @param newBackendName New backend.
 */
void copyRegisteredKernels(
    String registeredBackendName, String newBackendName) {
  final kernels = getKernelsForBackend(registeredBackendName);
  kernels.forEach((kernelConfig) {
    kernelConfig.backendName;
    final newKernelConfig = kernelConfig.withNewBackendName(newBackendName);
    registerKernel(newKernelConfig);
  });
}

String makeKey(String kernelName, String backendName) {
  return '${backendName}_${kernelName}';
}
