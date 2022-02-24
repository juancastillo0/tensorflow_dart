import 'package:tensorflow_wasm/src/ops/_prelude.dart';

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

// import {DataTypeMap, util} from '@tensorflow/tfjs-core';

import '../../util_base.dart' as util;

//DataTypeMap['float32' | 'int32']
List rangeImplCPU(
  int start,
  int stop,
  int step,
  // : 'float32'|'int32'
  DataType dtype,
) {
  final sameStartStop = start == stop;
  final increasingRangeNegativeStep = start < stop && step < 0;
  final decreasingRangePositiveStep = stop < start && step > 1;

  if (sameStartStop ||
      increasingRangeNegativeStep ||
      decreasingRangePositiveStep) {
    return util.makeZerosTypedArray(0, dtype);
  }

  final numElements = ((stop - start) / step).ceil().abs();
  final values = util.makeZerosTypedArray(numElements, dtype);

  if (stop < start && step == 1) {
    // Auto adjust the step's sign if it hasn't been set
    // (or was set to 1)
    step = -1;
  }

  values[0] = start;
  for (int i = 1; i < values.length; i++) {
    values[i] = values[i - 1] + step;
  }
  return values;
}
