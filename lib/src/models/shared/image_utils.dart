/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// import * as tf from '@tensorflow/tfjs-core';
// import {Matrix4x4} from './calculate_inverse_matrix';

// import {ImageSize, InputResolution, Padding, PixelInput, ValueTransform} from './interfaces/common_interfaces';
// import {Rect} from './interfaces/shape_interfaces';

import 'dart:math' as Math;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/src/util_base.dart' as util;

import 'calculate_inverse_matrix.dart';
import 'interfaces/common_interfaces.dart';
import 'interfaces/shape_interfaces.dart';

ImageSize getImageSize(PixelInput input) {
  if (input is tf.Tensor) {
    return ImageSize(
      height: input.shape[0].toDouble(),
      width: input.shape[1].toDouble(),
    );
  } else {
    return ImageSize(
        height: (input as dynamic).height, width: (input as dynamic).width);
  }
}

/**
 * Normalizes the provided angle to the range -pi to pi.
 * @param angle The angle in radians to be normalized.
 */
double normalizeRadians(double angle) {
  return angle - 2 * Math.pi * ((angle + Math.pi) / (2 * Math.pi)).floor();
}

/**
 * Transform value ranges.
 * @param fromMin Min of original value range.
 * @param fromMax Max of original value range.
 * @param toMin New min of transformed value range.
 * @param toMax New max of transformed value range.
 */
ValueTransform transformValueRange(
    double fromMin, double fromMax, double toMin, double toMax) {
  final fromRange = fromMax - fromMin;
  final toRange = toMax - toMin;

  if (fromRange == 0) {
    throw Exception(
        'Original min and max are both ${fromMin}, range cannot be 0.');
  }

  final scale = toRange / fromRange;
  final offset = toMin - fromMin * scale;
  return ValueTransform(
    scale: scale,
    offset: offset,
  );
}

/**
 * Convert an image to an image tensor representation.
 *
 * The image tensor has a shape [1, height, width, colorChannel].
 *
 * @param input An image, video frame, or image tensor.
 */
tf.Tensor toImageTensor(PixelInput input) {
  return input is tf.Tensor ? input : tf.browser.fromPixels(input);
}

class PaddedRoi {
  final Padding padding;
  final Rect roi;

  PaddedRoi({
    required this.padding,
    required this.roi,
  });
}

/**
 * Padding ratio of left, top, right, bottom, based on the output dimensions.
 *
 * The padding values are non-zero only when the "keep_aspect_ratio" is true.
 *
 * For instance, when the input image is 10x10 (width x height) and the
 * output dimensions is 20x40 and "keep_aspect_ratio" is true, we should scale
 * the input image to 20x20 and places it in the middle of the output image with
 * an equal padding of 10 pixels at the top and the bottom. The result is
 * therefore {left: 0, top: 0.25, right: 0, bottom: 0.25} (10/40 = 0.25f).
 * @param roi The original rectangle to pad.
 * @param targetSize The target width and height of the result rectangle.
 * @param keepAspectRatio Whether keep aspect ratio. Default to false.
 */
PaddedRoi padRoi(
  Rect roi,
  InputResolution targetSize, {
  bool keepAspectRatio = false,
}) {
  if (!keepAspectRatio) {
    return PaddedRoi(
      padding: Padding(top: 0, left: 0, right: 0, bottom: 0),
      roi: roi,
    );
  }

  final targetH = targetSize.height;
  final targetW = targetSize.width;

  validateSize(targetSize, 'targetSize');
  validateSize(roi, 'roi');

  final tensorAspectRatio = targetH / targetW;
  final roiAspectRatio = roi.height / roi.width;
  final double newWidth;
  final double newHeight;
  int horizontalPadding = 0;
  int verticalPadding = 0;
  if (tensorAspectRatio > roiAspectRatio) {
    // pad height;
    newWidth = roi.width;
    newHeight = roi.width * tensorAspectRatio;
    verticalPadding = ((1 - roiAspectRatio / tensorAspectRatio) / 2).round();
  } else {
    // pad width.
    newWidth = roi.height / tensorAspectRatio;
    newHeight = roi.height;
    horizontalPadding = ((1 - tensorAspectRatio / roiAspectRatio) / 2).round();
  }

  return PaddedRoi(
    padding: Padding(
      top: verticalPadding,
      left: horizontalPadding,
      right: horizontalPadding,
      bottom: verticalPadding,
    ),
    roi: roi.copyWith(
      height: newHeight,
      width: newWidth,
    ),
  );
}

/**
 * Get the rectangle information of an image, including xCenter, yCenter, width,
 * height and rotation.
 *
 * @param imageSize imageSize is used to calculate the rectangle.
 * @param normRect Optional. If normRect is not null, it will be used to get
 *     a subarea rectangle information in the image. `imageSize` is used to
 *     calculate the actual non-normalized coordinates.
 */
Rect getRoi(ImageSize imageSize, Rect? normRect) {
  if (normRect != null) {
    return Rect(
      xCenter: normRect.xCenter * imageSize.width,
      yCenter: normRect.yCenter * imageSize.height,
      width: normRect.width * imageSize.width,
      height: normRect.height * imageSize.height,
      rotation: normRect.rotation,
    );
  } else {
    return Rect(
      xCenter: 0.5 * imageSize.width,
      yCenter: 0.5 * imageSize.height,
      width: imageSize.width,
      height: imageSize.height,
      rotation: 0,
    );
  }
}

/**
 * Generate the projective transformation matrix to be used for `tf.transform`.
 *
 * See more documentation in `tf.transform`.
 *
 * @param matrix The transformation matrix mapping subRect to rect, can be
 *     computed using `getRotatedSubRectToRectTransformMatrix` calculator.
 * @param imageSize The original image height and width.
 * @param inputResolution The target height and width.
 */
List<double> getProjectiveTransformMatrix(
  Matrix4x4 matrix,
  ImageSize imageSize,
  InputResolution inputResolution,
) {
  validateSize(inputResolution, 'inputResolution');

  // To use M with regular x, y coordinates, we need to normalize them first.
  // Because x' = a0 * x + a1 * y + a2, y' = b0 * x + b1 * y + b2,
  // we need to use factor (1/inputResolution.width) to normalize x for a0 and
  // b0, similarly we need to use factor (1/inputResolution.height) to normalize
  // y for a1 and b1.
  // Also at the end, we need to de-normalize x' and y' to regular coordinates.
  // So we need to use factor imageSize.width for a0, a1 and a2, similarly
  // we need to use factor imageSize.height for b0, b1 and b2.
  final a0 = (1 / inputResolution.width) * matrix[0][0] * imageSize.width;
  final a1 = (1 / inputResolution.height) * matrix[0][1] * imageSize.width;
  final a2 = matrix[0][3] * imageSize.width;
  final b0 = (1 / inputResolution.width) * matrix[1][0] * imageSize.height;
  final b1 = (1 / inputResolution.height) * matrix[1][1] * imageSize.height;
  final b2 = matrix[1][3] * imageSize.height;

  return [a0, a1, a2, b0, b1, b2, 0, 0];
}

void validateSize(ImageSize size, String name) {
  util.assert_(size.width != 0, () => '${name} width cannot be 0.');
  util.assert_(size.height != 0, () => '${name} height cannot be 0.');
}
