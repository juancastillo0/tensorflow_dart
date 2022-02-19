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

import 'dart:typed_data';

import './src/util_base.dart';

// Utilities needed by backend consumers of tf-core.
// export './src/ops/axis_util.dart';
export './src/ops/broadcast_util.dart';
export './src/ops/concat_util.dart';
export './src/ops/conv_util.dart';
// export './src/ops/fused_util.dart';
// export './src/ops/fused_types.dart';
// export './src/ops/reduce_util.dart';

// import * as slice_util from './src/ops/slice_util.dart';
// export {slice_util};

// export {BackendValues, TypedArray, upcastType, PixelData} from './src/types.dart';
export './src/engine.dart' show MemoryInfo, TimingInfo;
// export './src/ops/rotate_util.dart';
// export './src/ops/array_ops_util.dart';
// export './src/ops/gather_nd_util.dart';
// export './src/ops/scatter_nd_util.dart';
// export './src/ops/selu_util.dart';
// export './src/ops/fused_util.dart';
// export './src/ops/erf_util.dart';
// export './src/log.dart';
// export './src/backends/complex_util.dart';
// export './src/backends/einsum_util.dart';
// export './src/ops/split_util.dart';
// export './src/ops/sparse/sparse_fill_empty_rows_util.dart';
// export './src/ops/sparse/sparse_reshape_util.dart';
// export './src/ops/sparse/sparse_segment_reduction_util.dart';

// import * as './src/ops/segment_util.dart';
// export {segment_util};

List<String> fromUint8ToStringArray(List<Uint8List> vals) {
  try {
    // Decode the bytes into string.
    return vals.map((val) => decodeString(val)).toList();
  } catch (err) {
    throw Exception(
        'Failed to decode encoded string bytes into utf-8, error: ${err}');
  }
}

List<Uint8List> fromStringArrayToUint8(List<String> strings) {
  return strings.map((s) => encodeString(s)).toList();
}
