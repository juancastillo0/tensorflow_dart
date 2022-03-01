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

import '../ops/buffer.dart';
import '../tensor.dart';

/** An implementation of the Where kernel shared between cpu and webgl */

// import {buffer} from '../ops/buffer';
// import {Tensor2D} from '../tensor';
// import {TypedArray} from '../types';

Tensor2D whereImpl(List<int> condShape, List condVals) {
  final indices = [];
  for (int i = 0; i < condVals.length; i++) {
    if (condVals[i]) {
      indices.add(i);
    }
  }

  final inBuffer = buffer(condShape, 'int32', null);

  final out = buffer([indices.length, condShape.length], 'int32', null);
  for (int i = 0; i < indices.length; i++) {
    final loc = inBuffer.indexToLoc(indices[i]);
    final offset = i * condShape.length;
    out.values.setAll(offset, loc);
  }
  return out.toTensor() as Tensor2D;
}
