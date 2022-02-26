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

// import {BackendWasm} from '../backend_wasm';

import 'dart:typed_data';

import '../../backend_wasm.dart';

// Analogous to `struct Result` in `non_max_suppression_impl.h`.
class Result {
  final int pSelectedIndices;
  final int selectedSize;
  final int pSelectedScores;
  final int pValidOutputs;

  Result({
    required this.pSelectedIndices,
    required this.selectedSize,
    required this.pSelectedScores,
    required this.pValidOutputs,
  });
}

/**
 * Parse the result of the c++ method, which has the shape equivalent to
 * `Result`.
 */
Result parseResultStruct(BackendWasm backend, int resOffset) {
  final result = Int32List.view(backend.wasm.HEAPU8.buffer, resOffset, 4);
  final pSelectedIndices = result[0];
  final selectedSize = result[1];
  final pSelectedScores = result[2];
  final pValidOutputs = result[3];
  // Since the result was allocated on the heap, we have to delete it.
  backend.wasm.free(resOffset);

  return Result(
    pSelectedIndices: pSelectedIndices,
    selectedSize: selectedSize,
    pSelectedScores: pSelectedScores,
    pValidOutputs: pValidOutputs,
  );
}
