import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/util_base.dart' show log;

import '../tensor_util.dart' show makeTypesMatch;
import '../tensor_util_env.dart' show convertToTensor;
import 'broadcast_util.dart' show assertAndGetBroadcastShape;

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
// import {ENGINE} from '../engine';
// import {isPromise} from '../util';

const OP_SCOPE_SUFFIX = '__op';

T execOp<T extends Tensors>(
  String name,
  T Function() fn,
  // String kernelName,
  // Map<String, Tensor> inputs, [
  // Map<String, Object>? attrs,]
) {
  String opName = name;
  // Strip the underscore from the end of the function name.
  if (opName.endsWith('_')) {
    opName = opName.substring(0, opName.length - 1);
  }

  // add an __op suffix to distinguish ops from kernels in tf.profile
  opName = opName + OP_SCOPE_SUFFIX;
  ENGINE.startScope(opName);
  try {
    final result = fn(); // ENGINE.runKernel(kernelName, inputs, attrs);
    if (result is Future) {
      log.severe('Cannot return a Promise inside of tidy.');
    }
    ENGINE.endScope(result);
    return result;
  } catch (ex) {
    ENGINE.endScope(null);
    rethrow;
  }
}

T execOpBinary<T extends Tensor>(
  String name,
  String kernelName,
  Tensor a,
  Tensor b, {
  String parseAsDtype = 'numeric',
}) {
  return execOp(name, () {
    var $a = convertToTensor(a, 'a', name, parseAsDtype);
    var $b = convertToTensor(b, 'b', name, parseAsDtype);
    final t = makeTypesMatch($a, $b);
    $a = t.first;
    $b = t.second;

    assertAndGetBroadcastShape($a.shape, $b.shape);

    final inputs = {'a': $a, 'b': $b}; // SquaredDifferenceInputs

    return ENGINE.runKernel(
      kernelName,
      inputs,
      {},
    ) as T;
  });
}

T execOpUnary<T extends Tensor>(
  String name,
  String kernelName,
  Tensor x, {
  String parseAsDtype = 'numeric',
}) {
  return execOp(name, () {
    final $x = convertToTensor(x, 'x', name, parseAsDtype);
    final inputs = {'x': $x};
    return ENGINE.runKernel(kernelName, inputs) as T;
  });
}

/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
dynamic Function(List) op<T>(Map<String, T Function(dynamic)> f) {
  final keys = f.keys.toList();
  if (keys.length != 1) {
    throw Exception('Please provide an object with a single key ' +
        '(operation name) mapping to a function. Got an object with ' +
        '${keys.length} keys.');
  }

  String opName = keys[0];
  final fn = f[opName]!;

  // Strip the underscore from the end of the function name.
  if (opName.endsWith('_')) {
    opName = opName.substring(0, opName.length - 1);
  }

  // add an __op suffix to distinguish ops from kernels in tf.profile
  opName = opName + OP_SCOPE_SUFFIX;

  // tslint:disable-next-line:no-any
  return (args) {
    ENGINE.startScope(opName);
    try {
      final result = fn(args);
      if (result is Future) {
        log.severe('Cannot return a Promise inside of tidy.');
      }
      ENGINE.endScope(result);
      return result;
    } catch (ex) {
      ENGINE.endScope(null);
      rethrow;
    }
  };
  // TODO: test
  // Object.defineProperty(f2, 'name', {value: opName, configurable: true});

  // tslint:disable-next-line:no-any
  // return f2 as T;
}
