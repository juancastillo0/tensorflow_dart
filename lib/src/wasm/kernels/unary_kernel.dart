import 'package:tensorflow_wasm/src/backend.dart';
import 'package:tensorflow_wasm/src/backend_wasm.dart';
import 'package:tensorflow_wasm/src/kernel_registry.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

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

// import {DataType, KernelConfig, TensorInfo, UnaryInputs, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {CppDType} from './types';

KernelConfig createUnaryKernelConfig(String kernelName, [DataType? outType]) {
  late Function(List)
      wasmFunc; //: (xId: number, dtype: number, outId: number) => void

  return KernelConfig(
    kernelName: kernelName,
    backendName: 'wasm',
    setupFunc: (Object backend) {
      wasmFunc =
          (backend as BackendWasm).wasm.cwrap(kernelName, null /* void */, [
        'number', // x_id
        'number', // dtype
        'number', // out_id
      ]);
    },
    kernelFunc: ({
      required Object backend,
      required NamedTensorInfoMap inputs, // TODO: UnaryInputs
      NamedAttrMap? attrs,
    }) {
      backend = backend as BackendWasm;

      final x = inputs['x']!;
      final xId = backend.dataIdMap.get(x.dataId)!.id;
      final out = backend.makeOutput(x.shape, outType ?? x.dtype);
      final outId = backend.dataIdMap.get(out.dataId)!.id;

      // Short-circuit zero-sized tensors.
      if (util.sizeFromShape(out.shape) == 0) {
        return ListOrVal.val(out);
      }

      wasmFunc([xId, CppDType.values.byName(x.dtype), outId]);
      return ListOrVal.val(out);
    },
  );
}

class KernelConfigG<B extends KernelBackend, A extends NamedAttrMap>
    implements KernelConfig {
  @override
  final String kernelName;
  @override
  final String backendName;
  final ListOrVal<TensorInfo> Function({
    required NamedTensorInfoMap inputs,
    required B backend,
    A? attrs,
  }) _kernelFunc;
  final void Function(B)? _setupFunc;
  final void Function(B)? _disposeFunc;

  @override
  // ignore: prefer_function_declarations_over_variables
  late final setupFunc = (Object backend) => _setupFunc?.call(backend as B);

  @override
  // ignore: prefer_function_declarations_over_variables
  late final disposeFunc = (Object backend) => _disposeFunc?.call(backend as B);

  @override
  // ignore: prefer_function_declarations_over_variables
  late final kernelFunc = ({
    required NamedTensorInfoMap inputs,
    required Object backend,
    NamedAttrMap? attrs,
  }) {
    return _kernelFunc(
      inputs: inputs,
      backend: backend as B,
      attrs: attrs as A?,
    );
  };

  KernelConfigG({
    required this.kernelName,
    required this.backendName,
    required ListOrVal<TensorInfo> Function({
      required NamedTensorInfoMap inputs,
      required B backend,
      A? attrs,
    })
        kernelFunc,
    void Function(B)? setupFunc,
    void Function(B)? disposeFunc,
  })  : _setupFunc = setupFunc,
        _disposeFunc = disposeFunc,
        _kernelFunc = kernelFunc;
}
