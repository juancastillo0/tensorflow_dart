// Required side effectful code for tfjs-core

// import {getOrMakeEngine} from './engine';
// import {buffer} from './ops/buffer';
// import {cast} from './ops/cast';
// import {clone} from './ops/clone';
// import {print} from './ops/print';
// import {OpHandler, setOpHandler} from './tensor';
import 'package:tensorflow_wasm/src/globals.dart' show deprecationWarn;
import 'package:tensorflow_wasm/src/ops/buffer.dart';
import 'package:tensorflow_wasm/src/ops/cast.dart';
import 'package:tensorflow_wasm/src/ops/clone.dart';
import 'package:tensorflow_wasm/src/ops/print.dart' as print_;

import 'tensor.dart' show OpHandler, setDeprecationWarningFn, setOpHandler;

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

// base.ts is tfjs-core without auto registration of things like flags,
// gradients, chained ops or the opHandler. See base_side_effects.ts for parts
// tfjs core that are required side effects.

/**
 * @fileoverview
 * @suppress {partialAlias} Optimization disabled due to passing the module
 * object into a function below:
 *
 *   import * as ops from './ops/ops';
 *   setOpHandler(ops);
 */

// Serialization.
// import * as io from './io/io';
// import * as math from './math';
// import './ops/broadcast_util.dart' as broadcast_util;
// import * as browser from './ops/browser';
// import * as gather_util from './ops/gather_nd_util';
// import * as scatter_util from './ops/scatter_nd_util';
// import * as slice_util from './ops/slice_util';
// import * as serialization from './serialization';
// import * as tensor_util from './tensor_util';
// import * as test_util from './test_util';
// import * as util from './util';
// import {version} from './version';

// export {InferenceModel, MetaGraph, MetaGraphInfo, ModelPredictConfig, ModelTensorInfo, SavedModelTensorInfo, SignatureDef, SignatureDefEntry, SignatureDefInfo} from './model_types';
// export {AdadeltaOptimizer} from './optimizers/adadelta_optimizer';
// export {AdagradOptimizer} from './optimizers/adagrad_optimizer';
// export {AdamOptimizer} from './optimizers/adam_optimizer';
// export {AdamaxOptimizer} from './optimizers/adamax_optimizer';
// export {MomentumOptimizer} from './optimizers/momentum_optimizer';
// export {Optimizer} from './optimizers/optimizer';
// // Optimizers.
// export {OptimizerConstructors} from './optimizers/optimizer_constructors';
// export {RMSPropOptimizer} from './optimizers/rmsprop_optimizer';
// export {SGDOptimizer} from './optimizers/sgd_optimizer';
export './tensor.dart'
    show
        DataToGPUOptions,
        DataToGPUWebGLOption,
        GPUData,
        // Scalar,
        Tensor,
        // Tensor1D,
        // Tensor2D,
        // Tensor3D,
        // Tensor4D,
        // Tensor5D,
        TensorBuffer,
        Variable,
        GradSaveFunc,
        NamedTensorMap,
        TensorContainer,
        TensorContainerArray,
        TensorContainerObject;
// export './tensor_types.dart'
//     show
//         GradSaveFunc,
//         NamedTensorMap,
//         TensorContainer,
//         TensorContainerArray,
//         TensorContainerObject;
// export {BackendValues, DataType, DataTypeMap, DataValues, NumericDataType, PixelData, Rank, RecursiveArray, ScalarLike, ShapeMap, sumOutType, TensorLike, TypedArray, upcastType} from './types';

export './ops/ops.dart';
// export {Reduction} from './ops/loss_ops_utils';

// export * from './train';
export './globals.dart';
export './kernel_registry.dart';
// export {customGrad, grad, grads, valueAndGrad, valueAndGrads, variableGrads} from './gradients';

export './engine.dart' show TimingInfo, MemoryInfo, ForwardFunc;
export './environment.dart' show Environment, env, ENV, TFPlatform;
// export {Platform} from './platforms/platform';

// export {version as version_core};

// // Top-level method exports.
// export {nextFrame} from './browser_util';

// // Second level exports.
// import * as backend_util from './backends/backend_util';
// import * as device_util from './device_util';
// export {
//   browser,
//   io,
//   math,
//   serialization,
//   test_util,
//   util,
//   backend_util,
//   broadcast_util,
//   tensor_util,
//   slice_util,
//   gather_util,
//   scatter_util,
//   device_util
// };

// import * as kernel_impls from './backends/kernel_impls';
// export {kernel_impls};
// // Backend specific.
export './backend.dart'
    show
        KernelBackend,
        BackendTimingInfo,
        DataMover,
        DataStorage; // TODO: was backends

// Export all kernel names / info.
export './kernel_names.dart';

void setUpOpHandler() {
  setDeprecationWarningFn(deprecationWarn);
  // Set up Engine and ENV
  // getOrMakeEngine();

  // Set up OpHandler
  final opHandler = OpHandler(buffer, cast, clone, print_.print_);
  setOpHandler(opHandler);
}

// // Register backend-agnostic flags.
// import './flags';
// // Register platforms
// import './platforms/platform_browser';
// import './platforms/platform_node';
