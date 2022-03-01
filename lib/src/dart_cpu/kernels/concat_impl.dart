import 'dart:typed_data';

import 'package:tensorflow_wasm/src/tensor.dart';

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

// import {backend_util, BackendValues, DataType, TypedArray, util} from '@tensorflow/tfjs-core';

import '../../util_base.dart' as util;
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

class ValueWithShape {
  final BackendValues vals;
  final Shape shape;

  ValueWithShape({
    required this.vals,
    required this.shape,
  });
}

List concatImplCPU(
  List<ValueWithShape> inputs,
  Shape outShape,
  DataType dtype,
  bool simplyConcat,
) {
  final outVals = util.getArrayFromDType(dtype, util.sizeFromShape(outShape));

  if (simplyConcat && dtype != 'string') {
    // Use built-in TypedArray.set() method for speed.
    int offset = 0;
    inputs.forEach((input) {
      final size = util.sizeFromShape(input.shape);

      outVals.setAll(offset, input.vals);
      offset += size;
    });
  } else {
    int colOffset = 0;

    inputs.forEach((input) {
      final decodedData = dtype == 'string'
          ? backend_util.fromUint8ToStringArray(input.vals as List<Uint8List>)
          : input.vals;

      int tIdx = 0;

      for (int row = 0; row < input.shape[0]; ++row) {
        final resIdx = row * outShape[1] + colOffset;
        for (int col = 0; col < input.shape[1]; ++col) {
          outVals[resIdx + col] = decodedData[tIdx++];
        }
      }

      colOffset += input.shape[1];
    });
  }

  return outVals;
}
