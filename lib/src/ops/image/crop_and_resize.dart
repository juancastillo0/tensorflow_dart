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
// import {CropAndResize, CropAndResizeAttrs, CropAndResizeInputs} from '../../kernel_names';
// import {NamedAttrMap} from '../../kernel_registry';
// import {Tensor1D, Tensor2D, Tensor4D} from '../../tensor';
// import {NamedTensorMap} from '../../tensor_types';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import * as util from '../../util';

// import {op} from '../operation';

import '../_prelude.dart';

import '../../util_base.dart' as util;

/**
 * Extracts crops from the input image tensor and resizes them using bilinear
 * sampling or nearest neighbor sampling (possibly with aspect ratio change)
 * to a common output size specified by cropSize.
 *
 * @param image 4d tensor of shape `[batch,imageHeight,imageWidth, depth]`,
 *     where imageHeight and imageWidth must be positive, specifying the
 *     batch of images from which to take crops
 * @param boxes 2d float32 tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the normalized
 *     coordinates of the box in the boxInd[i]'th image in the batch
 * @param boxInd 1d int32 tensor of shape `[numBoxes]` with values in range
 *     `[0, batch)` that specifies the image that the `i`-th box refers to.
 * @param cropSize 1d int32 tensor of 2 elements `[cropHeigh, cropWidth]`
 *     specifying the size to which all crops are resized to.
 * @param method Optional string from `'bilinear' | 'nearest'`,
 *     defaults to bilinear, which specifies the sampling method for resizing
 * @param extrapolationValue A threshold for deciding when to remove boxes based
 *     on score. Defaults to 0.
 * @return A 4D tensor of the shape `[numBoxes,cropHeight,cropWidth,depth]`
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
Tensor4D cropAndResize(
  Tensor4D image,
  Tensor2D boxes,
  Tensor1D boxInd,
  // : [number, number]
  List<int> cropSize, {
  // : 'bilinear'|'nearest'
  String method = 'bilinear',
  double extrapolationValue = 0,
}) {
  return execOp('cropAndResize', () {
    final $image = convertToTensor(image, 'image', 'cropAndResize');
    final $boxes = convertToTensor(boxes, 'boxes', 'cropAndResize', 'float32');
    final $boxInd = convertToTensor(boxInd, 'boxInd', 'cropAndResize', 'int32');

    final numBoxes = $boxes.shape[0];

    util.assert_(
        $image.rank == 4,
        () =>
            'Error in cropAndResize: image must be rank 4,' +
            'but got rank ${$image.rank}.');
    util.assert_(
        $boxes.rank == 2 && $boxes.shape[1] == 4,
        () =>
            'Error in cropAndResize: boxes must be have size [${numBoxes},4] ' +
            'but had shape ${$boxes.shape}.');
    util.assert_(
        $boxInd.rank == 1 && $boxInd.shape[0] == numBoxes,
        () =>
            'Error in cropAndResize: boxInd must be have size [${numBoxes}] ' +
            'but had shape ${$boxes.shape}.');
    util.assert_(
        cropSize.length == 2,
        () =>
            'Error in cropAndResize: cropSize must be of length 2, but got ' +
            'length ${cropSize.length}.');
    util.assert_(cropSize[0] >= 1 && cropSize[1] >= 1,
        () => 'cropSize must be atleast [1,1], but was ${cropSize}');
    util.assert_(method == 'bilinear' || method == 'nearest',
        () => 'method must be bilinear or nearest, but was ${method}');

    final inputs = {
      'image': $image,
      'boxes': $boxes,
      'boxInd': $boxInd
    }; // CropAndResizeInputs
    final attrs = {
      'method': method,
      'extrapolationValue': extrapolationValue,
      'cropSize': cropSize,
    }; // CropAndResizeAttrs
    final res = ENGINE.runKernel(CropAndResize, inputs, attrs);
    return res as Tensor4D;
  });
}
