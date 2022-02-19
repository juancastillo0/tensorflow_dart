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

// import {assert} from '../util';

// import {OnProgressCallback} from './types';

import 'package:tensorflow_wasm/src/io/types.dart';

/**
 * Monitor Promise.all progress, fire onProgress callback function.
 *
 * @param promises Promise list going to be monitored
 * @param onProgress Callback function. Fired when a promise resolved.
 * @param startFraction Optional fraction start. Default to 0.
 * @param endFraction Optional fraction end. Default to 1.
 */
Future<List<T>> monitorPromisesProgress<T>(
  List<Future<T>> promises,
  OnProgressCallback onProgress, [
  double startFraction = 0,
  double endFraction = 1,
]) {
  _checkPromises(promises);
  _checkFraction(startFraction, endFraction);
  int resolvedPromise = 0;

  Future<T> registerMonitor(Future<T> promise) {
    promise.then((value) {
      final fraction = startFraction +
          ++resolvedPromise / promises.length * (endFraction - startFraction);
      // pass fraction as parameter to callback function.
      onProgress(fraction);
      return value;
    });
    return promise;
  }

  return Future.wait(promises.map(registerMonitor));
}

void _checkFraction(double startFraction, double endFraction) {
  assert(
      startFraction >= 0 && startFraction <= 1,
      () =>
          'Progress fraction must be in range [0, 1], but ' +
          'got startFraction ${startFraction}');
  assert(
      endFraction >= 0 && endFraction <= 1,
      () =>
          'Progress fraction must be in range [0, 1], but ' +
          'got endFraction ${endFraction}');
  assert(
      endFraction >= startFraction,
      () =>
          'startFraction must be no more than endFraction, but ' +
          'got startFraction ${startFraction} and endFraction ' +
          '${endFraction}');
}

void _checkPromises(List<Future> promises) {
  assert(promises != null && promises is List && promises.length > 0,
      () => 'promises must be a none empty array');
}
