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

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/backend_wasm.dart';
import 'package:tensorflow_wasm/src/kernel_registry.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:tensorflow_wasm/src/ops/broadcast_util.dart' as broadcast_util;

// import {backend_util, BinaryInputs, DataType, KernelConfig, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

typedef WasmTFFn = void Function(int aId, Uint8List aShape, int aShapeLen,
    int bId, Uint8List bShape, int bShapeLen, int dtype, int outId);

KernelConfig createBinaryKernelConfig(
  String kernelName, {
  required bool supportsFullBroadcast,
  DataType? dtype,
}) {
  late Function(List) wasmFunc;

  void setupFunc(Object backend) {
    wasmFunc =
        (backend as BackendWasm).wasm.cwrap(kernelName, null /* void */, [
      'number', // a_id,
      'array', // a_shape
      'number', // a_shape.length
      'number', // b_id
      'array', // b_shape
      'number', // b_shape.length
      'number', // dtype
      'number' // out_id
    ]);
  }

  ListOrVal<TensorInfo> kernelFunc({
    required Map<String, TensorInfo> inputs,
    required Object backend,
    Map<String, Object>? attrs,
  }) {
    // TODO:
    // inputs = inputs as BinaryInputs;
    backend = backend as BackendWasm;
    final a = inputs['a']!;
    final b = inputs['b']!;
    final aId = backend.dataIdMap.get(a.dataId)!.id;
    final bId = backend.dataIdMap.get(b.dataId)!.id;

    final outputType = dtype != null ? dtype : a.dtype;
    final newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    final out = backend.makeOutput(newShape, outputType);

    // Short-circuit zero-sized tensors.
    if (util.sizeFromShape(newShape) == 0) {
      return ListOrVal.val(out);
    }

    final aShapeBytes = Uint8List.view(Int32List.fromList(a.shape).buffer);
    final bShapeBytes = Uint8List.view(Int32List.fromList(b.shape).buffer);
    final outId = backend.dataIdMap.get(out.dataId)!.id;
    // ignore: prefer_function_declarations_over_variables
    final kernelFunc = () => wasmFunc([
          aId,
          aShapeBytes,
          a.shape.length,
          bId,
          bShapeBytes,
          b.shape.length,
          CppDType.values.byName(a.dtype).index,
          outId
        ]);

    kernelFunc();
    return ListOrVal.val(out);
  }

  return KernelConfig(
    kernelName: kernelName,
    backendName: 'wasm',
    setupFunc: setupFunc,
    kernelFunc: kernelFunc,
  );
}
