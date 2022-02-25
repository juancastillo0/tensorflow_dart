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

// import {ENGINE} from '../engine';
// import {DepthToSpace, DepthToSpaceAttrs, DepthToSpaceInputs} from '../kernel_names';
// import {NamedAttrMap} from '../kernel_registry';
// import {Tensor4D} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {TensorLike4D} from '../types';
// import * as util from '../util';

// import {op} from './operation';

import '_prelude.dart';
import '../util_base.dart' as util;

/**
 * Rearranges data from depth into blocks of spatial data. More specifically,
 * this op outputs a copy of the input tensor where values from the `depth`
 * dimension are moved in spatial blocks to the `height` and `width` dimensions.
 * The attr `blockSize` indicates the input block size and how the data is
 * moved.
 *
 *  - Chunks of data of size `blockSize * blockSize` from depth are rearranged
 * into non-overlapping blocks of size `blockSize x blockSize`
 *
 *  - The width the output tensor is `inputWidth * blockSize`, whereas the
 * height is `inputHeight * blockSize`
 *
 *  - The Y, X coordinates within each block of the output image are determined
 * by the high order component of the input channel index
 *
 *  - The depth of the input tensor must be divisible by `blockSize *
 * blockSize`
 *
 * The `dataFormat` attr specifies the layout of the input and output tensors
 * with the following options: "NHWC": [ `batch, height, width, channels` ]
 * "NCHW": [ `batch, channels, height, width` ]
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
 * const blockSize = 2;
 * const dataFormat = "NHWC";
 *
 * tf.depthToSpace(x, blockSize, dataFormat).print();
 * ```
 *
 * @param x The input tensor of rank 4
 * @param blockSIze  An `int` that is `>= 2`. The size of the spatial block
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to "NHWC"
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
Tensor4D depthToSpace(
  Tensor4D x,
  int blockSize, {
  String dataFormat = 'NHWC', //  'NHWC'|'NCHW'
}) {
  return execOp('depthToSpace', () {
    final $x = convertToTensor(x, 'x', 'depthToSpace', 'float32') as Tensor4D;

    final inputHeight = (dataFormat == 'NHWC') ? $x.shape[1] : $x.shape[2];
    final inputWidth = (dataFormat == 'NHWC') ? $x.shape[2] : $x.shape[3];
    final inputDepth = (dataFormat == 'NHWC') ? $x.shape[3] : $x.shape[1];

    util.assert_(
        blockSize > 1,
        () =>
            'blockSize should be > 1 for depthToSpace, but was: ${blockSize}');

    util.assert_(
        inputHeight * blockSize >= 0,
        () => 'Negative dimension size caused by overflow when multiplying'
            '${inputHeight} and ${blockSize}  for depthToSpace with input shape ${$x.shape}');

    util.assert_(
        inputWidth * blockSize >= 0,
        () => 'Negative dimension size caused by overflow when multiplying'
            '${inputWidth} and ${blockSize} for '
            'depthToSpace with input shape ${$x.shape}');

    util.assert_(
        (inputDepth % (blockSize * blockSize) == 0),
        () =>
            'Dimension size must be evenly divisible by ${blockSize * blockSize} but is ${inputDepth} for depthToSpace with input shape ${$x.shape}');

    final inputs = {'x': $x}; // : DepthToSpaceInputs
    final attrs = {
      'blockSize': blockSize,
      'dataFormat': dataFormat,
    }; // : DepthToSpaceAttrs

    return ENGINE.runKernel(DepthToSpace, inputs, attrs) as Tensor4D;
  });
}
