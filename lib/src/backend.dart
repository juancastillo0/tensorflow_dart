import 'package:tensorflow_wasm/src/tensor.dart';

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// import {Backend, DataId, DataToGPUOptions, GPUData} from '../tensor';
// import {BackendValues, DataType} from '../types';

const EPSILON_FLOAT32 = 1e-7;
const EPSILON_FLOAT16 = 1e-4;

// Required information for all backends.
class BackendTimingInfo {
  // a field for additional timing information
  final Object kernelMs; // number|{error: string};
  // e.g. packing / unpacking for WebGL backend
  final String Function()? getExtraProfileInfo;

  BackendTimingInfo(this.kernelMs, {this.getExtraProfileInfo});
}

class MemoryInfo {
  final bool unreliable;
  final List<String>? reasons;

  MemoryInfo({
    required this.unreliable,
    this.reasons,
  });
}

abstract class TensorStorage {
  Future<BackendValues> read(DataId dataId);
  BackendValues readSync(DataId dataId);
  bool disposeData(DataId dataId, {bool force});
  DataId write(BackendValues values, List<int> shape, DataType dtype);
  void move(DataId dataId, BackendValues values, List<int> shape,
      DataType dtype, int refCount);
  MemoryInfo memory(); // Backend-specific information.
  /** Returns number of data ids currently in the storage. */
  int numDataIds();
  int refCount(DataId dataId);
}

/** Convenient class for storing tensor-related data. */
class DataStorage<T extends Object> {
  // TODO: WeakMap
  final data = <DataId, T>{};
  int dataIdsCount = 0;
  final KernelBackend backend;
  final DataMover dataMover;

  DataStorage(this.backend, this.dataMover);

  T? get(DataId dataId) {
    if (!this.data.containsKey(dataId)) {
      this.dataMover.moveData(this.backend, dataId);
    }
    return this.data[dataId];
  }

  void set(DataId dataId, T value) {
    this.dataIdsCount++;
    this.data[dataId] = value;
  }

  bool has(DataId dataId) {
    return this.data.containsKey(dataId);
  }

  bool delete(DataId dataId) {
    this.dataIdsCount--;
    return this.data.remove(dataId) != null;
  }

  int numDataIds() {
    return this.dataIdsCount;
  }
}

abstract class DataMover {
  /**
   * To be called by backends whenever they see a dataId that they don't own.
   * Upon calling this method, the mover will fetch the tensor from another
   * backend and register it with the current active backend.
   */
  void moveData(KernelBackend backend, DataId dataId);
}

abstract class BackendTimer {
  // check if backend timer is available
  bool timerAvailable();
  Future<BackendTimingInfo> time(void Function() f);
}

/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
class KernelBackend implements TensorStorage, Backend, BackendTimer {
  int refCount(DataId dataId) {
    return notYetImplemented('refCount');
  }

  void incRef(DataId dataId) {
    notYetImplemented('incRef');
  }

  bool timerAvailable() {
    return true;
  }

  Future<BackendTimingInfo> time(void Function() f) {
    return notYetImplemented('time');
  }

  Future<BackendValues> read(DataId dataId) {
    return notYetImplemented('read');
  }

  BackendValues readSync(DataId dataId) {
    return notYetImplemented('readSync');
  }

  GPUData readToGPU(DataId dataId, {DataToGPUOptions? options}) {
    return notYetImplemented('readToGPU');
  }

  int numDataIds() {
    return notYetImplemented('numDataIds');
  }

  bool disposeData(DataId dataId, {bool force = false}) {
    return notYetImplemented('disposeData');
  }

  DataId write(BackendValues values, List<int> shape, DataType dtype) {
    return notYetImplemented('write');
  }

  void move(
    DataId dataId,
    BackendValues values,
    List<int> shape,
    DataType dtype,
    int refCount,
  ) {
    notYetImplemented('move');
  }

  MemoryInfo memory() {
    return notYetImplemented('memory');
  }

  /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
  int floatPrecision() {
    return notYetImplemented('floatPrecision');
  }

  /** Returns the smallest representable number.  */
  double epsilon() {
    return this.floatPrecision() == 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  }

  void dispose() {
    notYetImplemented('dispose');
  }
}

Never notYetImplemented(String kernelName) {
  throw Exception(
      "'${kernelName}' not yet implemented or not found in the registry. " +
          'This kernel may not be supported by the tfjs backend you have chosen');
}
