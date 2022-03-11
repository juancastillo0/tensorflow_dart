/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// import * as tf from '@tensorflow/tfjs-core';
// import {TUNABLE_FLAG_VALUE_RANGE_MAP} from './params';
import 'dart:html';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'params.dart';

bool isiOS() {
  return RegExp('iPhone|iPad|iPod', caseSensitive: false)
      .hasMatch(window.navigator.userAgent);
}

bool isAndroid() {
  return RegExp('Android', caseSensitive: false)
      .hasMatch(window.navigator.userAgent);
}

bool isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Reset the target backend.
 *
 * @param backendName The name of the backend to be reset.
 */
Future<void> resetBackend(String backendName) async {
  final ENGINE = tf.engine();
  if (!ENGINE.registryFactory.containsKey(backendName)) {
    throw Exception('${backendName} backend is not registed.');
  }
  final backendFactory = tf.findBackendFactory(backendName);
  if (backendFactory == null) {
    return;
  }
  final backend = tf.BackendFactory(
    backendName,
    backendFactory,
  );
  if (ENGINE.registry.containsKey(backendName)) {
    tf.removeBackend(backendName);
    tf.registerBackend(backend);
  }

  await tf.setBackend(backend);
}

/**
 * Set environment flags.
 *
 * This is a wrapper function of `tf.env().setFlags()` to constrain users to
 * only set tunable flags (the keys of `TUNABLE_FLAG_TYPE_MAP`).
 *
 * ```js
 * const flagConfig = {
 *        WEBGL_PACK: false,
 *      };
 * await setEnvFlags(flagConfig);
 *
 * console.log(tf.env().getBool('WEBGL_PACK')); // false
 * console.log(tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')); // false
 * ```
 *
 * @param flagConfig An object to store flag-value pairs.
 */
Future<void> setBackendAndEnvFlags(
    Map<String, Object?> flagConfig, String backend) async {
  // if (flagConfig == null) {
  //   return;
  // } else if (typeof flagConfig != 'object') {
  //   throw Exception(
  //       "An object is expected, while a(n) ${flagConfig.runtimeType} is found.");
  // }

  // Check the validation of flags and values.
  for (final flag in flagConfig.keys) {
    // TODO: check whether flag can be set as flagConfig[flag].
    if (!TUNABLE_FLAG_VALUE_RANGE_MAP.containsKey(flag)) {
      throw Exception("${flag} is not a tunable or valid environment flag.");
    }
    if (TUNABLE_FLAG_VALUE_RANGE_MAP[flag]!.indexOf(flagConfig[flag]!) == -1) {
      throw Exception(
          "${flag} value is expected to be in the range [${TUNABLE_FLAG_VALUE_RANGE_MAP[flag]}], while ${flagConfig[flag]}" +
              ' is found.');
    }
  }

  tf.env().setFlags(Map.fromEntries(
        flagConfig.entries.where((element) => element.value != null).cast(),
      ));

  final backendSplit = backend.split('-');

  if (backendSplit[0] == 'tfjs') {
    await resetBackend(backendSplit[1]);
  }
}
