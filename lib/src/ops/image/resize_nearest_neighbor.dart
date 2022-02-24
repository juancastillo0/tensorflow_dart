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
// import {ResizeNearestNeighbor, ResizeNearestNeighborAttrs, ResizeNearestNeighborInputs} from '../../kernel_names';
// import {NamedAttrMap} from '../../kernel_registry';
// import {Tensor3D, Tensor4D} from '../../tensor';
// import {NamedTensorMap} from '../../tensor_types';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import * as util from '../../util';

// import {op} from '../operation';
// import {reshape} from '../reshape';

import '../_prelude.dart';

import '../reshape.dart' show reshape;
import '../../util_base.dart' as util;

/**
 * NearestNeighbor resize a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to False. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 * @param halfPixelCenters Defaults to `false`. Whether to assumes pixels are of
 *      half the actual dimensions, and yields more accurate resizes. This flag
 *      would also make the floating point coordinates of the top left pixel
 *      0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
T resizeNearestNeighbor<
    T extends Tensor3D
// |Tensor4D
    >(
  T images,
  // : [number, number]
  List<int> size, {
  bool alignCorners = false,
  bool halfPixelCenters = false,
}) {
  return execOp('resizeNearestNeighbor', () {
    final $images = convertToTensor(images, 'images', 'resizeNearestNeighbor');

    util.assert_(
        $images.rank == 3 || $images.rank == 4,
        () =>
            'Error in resizeNearestNeighbor: x must be rank 3 or 4, but got ' +
            'rank ${$images.rank}.');
    util.assert_(
        size.length == 2,
        () =>
            'Error in resizeNearestNeighbor: new shape must 2D, but got shape ' +
            '${size}.');
    util.assert_($images.dtype == 'float32' || $images.dtype == 'int32',
        () => '`images` must have `int32` or `float32` as dtype');
    util.assert_(
        halfPixelCenters == false || alignCorners == false,
        () =>
            'Error in resizeNearestNeighbor: If halfPixelCenters is true, ' +
            'alignCorners must be false.');
    Tensor4D batchImages = $images as Tensor4D;
    bool reshapedTo4D = false;
    if ($images.rank == 3) {
      reshapedTo4D = true;
      batchImages = reshape(
          $images, [1, $images.shape[0], $images.shape[1], $images.shape[2]]);
    }

    final inputs = {'images': batchImages}; // ResizeNearestNeighborInputs
    final attrs = {
      'alignCorners': alignCorners,
      'halfPixelCenters': halfPixelCenters,
      'size': size,
    }; // ResizeNearestNeighborAttrs

    // tslint:disable-next-line: no-unnecessary-type-assertion
    final res = ENGINE.runKernel(ResizeNearestNeighbor, inputs, attrs) as T;

    if (reshapedTo4D) {
      return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
    }
    return res;
  });
}
