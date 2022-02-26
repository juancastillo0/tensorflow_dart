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

// import {ENGINE} from '../../engine';
// import {FlipLeftRight, FlipLeftRightInputs} from '../../kernel_names';
// import {Tensor4D} from '../../tensor';
// import {NamedTensorMap} from '../../tensor_types';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import * as util from '../../util';
// import {op} from '../operation';

import '../_prelude.dart';
import '../../util_base.dart' as util;

/**
 * Flips the image left to right. Currently available in the CPU, WebGL, and
 * WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 */
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
Tensor4D flipLeftRight(Tensor4D image) {
  return execOp('flipLeftRight', () {
    final $image = convertToTensor(image, 'image', 'flipLeftRight', 'float32');

    util.assert_(
        $image.rank == 4,
        () =>
            'Error in flipLeftRight: image must be rank 4,' +
            'but got rank ${$image.rank}.');

    final inputs = {'image': $image}; //: FlipLeftRightInputs
    final res = ENGINE.runKernel(FlipLeftRight, inputs, {});
    return res as Tensor4D;
  });
}
