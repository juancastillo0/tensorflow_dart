import 'package:tensorflow_wasm/src/backend.dart';
import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

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

// import {BackendTimer, BackendTimingInfo} from './backends/backend';
// import {env} from './environment';
// import {Tensor} from './tensor';
// import {NamedTensorMap} from './tensor_types';
// import {DataType, DataTypeMap, TypedArray} from './types';
// import * as util from './util';

class KernelProfile {
  final String kernelName;
  final List<Tensor> outputs;
  final NamedTensorMap inputs;
  final Future<Object> timeMs; // number|{error: String}
  final Future<String> extraInfo;

  KernelProfile({
    required this.kernelName,
    required this.outputs,
    required this.inputs,
    required this.timeMs,
    required this.extraInfo,
  });
}

class Profiler {
  final ProfileLogger logger;
  final BackendTimer backendTimer;

  Profiler(this.backendTimer, {ProfileLogger? logger})
      : logger = logger ?? ProfileLogger();

  KernelProfile profileKernel(
      String kernelName, NamedTensorMap inputs, List<Tensor> Function() f) {
    late List<Tensor> outputs;
    final holdResultWrapperFn = () {
      outputs = f();
    };
    Future<BackendTimingInfo> timer;
    final start = util.now();
    if (this.backendTimer.timerAvailable()) {
      timer = this.backendTimer.time(holdResultWrapperFn);
    } else {
      holdResultWrapperFn();
      for (final output in outputs) {
        output.dataSync();
      }
      timer = Future.value(BackendTimingInfo(util.now() - start));
    }
    if (env().getBool('CHECK_COMPUTATION_FOR_ERRORS')) {
      for (int i = 0; i < outputs.length; i++) {
        final output = outputs[i];
        // Dangling promise here because we don't want to propagate up
        // asynchronicity.
        output.data().then((tensorVals) {
          checkComputationForErrors(tensorVals, output.dtype, kernelName);
        });
      }
    }

    final kernelProfile = KernelProfile(
      kernelName: kernelName,
      outputs: outputs,
      inputs: inputs,
      timeMs: timer.then((timing) => timing.kernelMs),
      extraInfo: timer.then((timing) => timing.getExtraProfileInfo != null
          ? timing.getExtraProfileInfo!()
          : ''),
    );
    return kernelProfile;
  }

  void logKernelProfile(KernelProfile kernelProfile) {
    kernelProfile.outputs.forEach((result) {
      Future.wait([
        result.data(),
        kernelProfile.timeMs,
        kernelProfile.extraInfo,
      ]).then((valueContainer) {
        this.logger.logKernelProfile(
              kernelProfile.kernelName,
              result,
              valueContainer[0] as List,
              valueContainer[1],
              kernelProfile.inputs,
              valueContainer[2] as String,
            );
      });
    });
  }
}

bool checkComputationForErrors<D extends DataType>(
  // DataTypeMap[D]
  Object vals,
  D dtype,
  String kernelName,
) {
  if (dtype != 'float32') {
    // Only floating point computations will generate NaN values
    return false;
  }
  for (int i = 0; i < (vals as List).length; i++) {
    final num_ = vals[i] as num;
    if (num_.isNaN || !num_.isFinite) {
      // Throwing custom exception so behavior is testable.
      util.log.warning("Found ${num_} in the result of '${kernelName}'");
      return true;
    }
  }
  return false;
}

class ProfileLogger {
  logKernelProfile(
    String name,
    Tensor result,
    List vals,
    // number|{error: string}
    Object timeMs,
    NamedTensorMap inputs,
    String? extraInfo,
  ) {
    final time = timeMs is num
        ? util.rightPad('${timeMs}ms', 9)
        : (timeMs as Map)['error'];
    final paddedName = util.rightPad(name, 25);
    final rank = result.rank;
    final size = result.size;
    final shape = util.rightPad(result.shape.toString(), 14);
    String inputShapesDescription = '';

    for (final name in inputs.keys) {
      final input = inputs[name];
      if (input != null) {
        // The input might be a non-tensor (e.g HTMLImageElement), in which case
        // we claim the output shape as input shape.
        final inputShape = input.shape ?? result.shape;
        final inputRank = inputShape.length;
        inputShapesDescription +=
            '${name}: ${inputRank}D ${inputRank > 0 ? inputShape : ''} ';
      }
    }

    util.log.info(
      '%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}\t%c${inputShapesDescription}\t%c${extraInfo}',
    );
  }
}
