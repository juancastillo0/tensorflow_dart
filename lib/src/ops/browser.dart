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

// import {ENGINE} from '../engine';
// import {env} from '../environment';
// import {FromPixels, FromPixelsAttrs, FromPixelsInputs} from '../kernel_names';
// import {getKernel, NamedAttrMap} from '../kernel_registry';
// import {Tensor, Tensor2D, Tensor3D} from '../tensor';
// import {NamedTensorMap} from '../tensor_types';
// import {convertToTensor} from '../tensor_util_env';
// import {PixelData, TensorLike} from '../types';

// import {cast} from './cast';
// import {op} from './operation';
// import {tensor3d} from './tensor3d';

import 'dart:typed_data';
import 'package:universal_html/html.dart';

import '../environment.dart';
import '../js/js_api.dart';
// import '../kernel_registry.dart' show getKernel;
import '../types.dart';
import '_prelude.dart';
import 'cast.dart';
import 'tensor.dart';

CanvasRenderingContext2D? fromPixels2DContext;

/**
 * Creates a `tf.Tensor` from an image.
 *
 * ```js
 * final image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * tf.browser.fromPixels(image).print();
 * ```
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8List; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @returns A Tensor3D with the shape `[height, width, numChannels]`.
 *
 * Note: fromPixels can be lossy in some cases, same image may result in
 * slightly different tensor values, if rendered by different rendering
 * engines. This means that results from different browsers, or even same
 * browser with CPU and GPU rendering engines can be different. See discussion
 * in details:
 * https://github.com/tensorflow/tfjs/issues/5482
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
Tensor3D fromPixels(Object pixels,
    // : PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
    // HTMLVideoElement|ImageBitmap,
    {int numChannels = 3}) {
  return execOp('fromPixels', () {
    // Sanity checks.
    if (numChannels > 4) {
      throw Exception(
          'Cannot construct Tensor with more than 4 channels from pixels.');
    }
    if (pixels == null) {
      throw Exception(
          'pixels passed to tf.browser.fromPixels() can not be null');
    }
    bool isPixelData = false;
    bool isImageData = false;
    bool isVideo = false;
    bool isImage = false;
    bool isCanvasLike = false;
    bool isImageBitmap = false;

    if (pixels is PixelData) {
      isPixelData = true;
    } else if (pixels is ImageData) {
      isImageData = true;
    } else if (pixels is VideoElement) {
      isVideo = true;
    } else if (pixels is ImageElement) {
      isImage = true;
    } else if (pixels is CanvasElement) {
      isCanvasLike = true;
    } else if (pixels is ImageBitmap) {
      isImageBitmap = true;
    } else {
      throw Exception(
          'pixels passed to tf.browser.fromPixels() must be either an ' +
              "HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData " +
              "in browser, or OffscreenCanvas, ImageData in webworker" +
              " or {data: Uint32List, width: number, height: number}, " +
              "but was ${pixels.runtimeType}");
    }
    if (isVideo) {
      final HAVE_CURRENT_DATA_READY_STATE = 2;
      if (isVideo &&
          (pixels as VideoElement).readyState < HAVE_CURRENT_DATA_READY_STATE) {
        throw Exception(
            'The video element has not loaded data yet. Please wait for ' +
                '`loadeddata` event on the <video> element.');
      }
    }
    // If the current backend has 'FromPixels' registered, it has a more
    // efficient way of handling pixel uploads, so we call that.
    // final kernel = getKernel(FromPixels, ENGINE.backendName!);
    // if (kernel != null) {
    //   final inputs = {'pixels': pixels}; // : FromPixelsInputs
    //   final attrs = {'numChannels': numChannels}; // : FromPixelsAttrs
    //   return ENGINE.runKernel(
    //       FromPixels, inputs,
    //       attrs);
    // }

    final int width, height;
    if (pixels is VideoElement) {
      width = pixels.videoWidth;
      height = pixels.videoHeight;
    } else {
      width = (pixels as dynamic).width;
      height = (pixels as dynamic).height;
    }
    final TypedData vals;

    if (pixels is CanvasElement) {
      vals =
          // tslint:disable-next-line:no-any
          (pixels.getContext('2d') as CanvasRenderingContext2D)
              .getImageData(0, 0, width, height)
              .data;
    } else if (pixels is PixelData) {
      vals = pixels.data;
    } else if (pixels is ImageData) {
      vals = pixels.data;
    } else if (isImage || isVideo || isImageBitmap) {
      // ignore: prefer_conditional_assignment
      if (fromPixels2DContext == null) {
        // TODO:
        // if (typeof document == 'undefined') {
        //   if (typeof OffscreenCanvas != 'undefined' &&
        //       typeof OffscreenCanvasRenderingContext2D != 'undefined') {
        //     // @ts-ignore
        //     fromPixels2DContext = new OffscreenCanvas(1, 1).getContext('2d');
        //   } else {
        //     throw Exception(
        //         'Cannot parse input in current context. ' +
        //         'Reason: OffscreenCanvas Context2D rendering is not supported.');
        //   }
        // } else {
        fromPixels2DContext =
            (document.createElement('canvas') as CanvasElement).getContext('2d')
                as CanvasRenderingContext2D;
        // }
      }
      final _fromPixels2DContext = fromPixels2DContext!;
      _fromPixels2DContext.canvas.width = width;
      _fromPixels2DContext.canvas.height = height;
      _fromPixels2DContext.drawImageScaled(
          pixels as VideoElement, 0, 0, width, height);
      vals = _fromPixels2DContext.getImageData(0, 0, width, height).data;
    } else {
      throw Error();
    }

    final Int32List values;
    if (numChannels == 4) {
      values = Int32List.sublistView(vals);
    } else {
      final numPixels = width * height;
      values = Int32List(numPixels * numChannels);
      for (int i = 0; i < numPixels; i++) {
        for (int channel = 0; channel < numChannels; ++channel) {
          values[i * numChannels + channel] =
              (vals as List<int>)[i * 4 + channel];
        }
      }
    }
    final outShape = [height, width, numChannels];
    return tensor(values, outShape, 'int32');
  });
}

// Helper functions for |fromPixelsAsync| to check whether the input can
// be wrapped into imageBitmap.
bool _isPixelData(Object pixels
// : PixelData|ImageData|HTMLImageElement|
//                      HTMLCanvasElement|HTMLVideoElement|
//                      ImageBitmap
    ) {
  return (pixels != null) && ((pixels as PixelData).data is Uint8List);
}

bool _isNonEmptyPixels(Object pixels
// : PixelData|ImageData|HTMLImageElement|
//                           HTMLCanvasElement|HTMLVideoElement|ImageBitmap
    ) {
  return pixels != null &&
      (pixels as dynamic).width != 0 &&
      (pixels as dynamic).height != 0;
}

bool _canWrapPixelsToImageBitmap(Object pixels
// : PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
//                                     HTMLVideoElement|ImageBitmap
    ) {
  return isImageBitmapFullySupported() &&
      !(pixels is ImageBitmap) &&
      _isNonEmptyPixels(pixels) &&
      !_isPixelData(pixels);
}

/**
 * Creates a `tf.Tensor` from an image in async way.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * (await tf.browser.fromPixelsAsync(image)).print();
 * ```
 * This API is the async version of fromPixels. The API will first
 * check |WRAP_TO_IMAGEBITMAP| flag, and try to wrap the input to
 * imageBitmap if the flag is set to true.
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8List; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
Future fromPixelsAsync(
  Object pixels,
  // : PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
  // HTMLVideoElement|ImageBitmap,
  {
  int numChannels = 3,
}) async {
  Object? inputs;
  // : PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
  //     HTMLVideoElement|ImageBitmap = null;

  // Check whether the backend needs to wrap |pixels| to imageBitmap and
  // whether |pixels| can be wrapped to imageBitmap.
  if (env().getBool('WRAP_TO_IMAGEBITMAP') &&
      _canWrapPixelsToImageBitmap(pixels)) {
    // Force the imageBitmap creation to not do any premultiply alpha
    // ops.
    ImageBitmap? imageBitmap;

    try {
      // wrap in try-catch block, because createImageBitmap may not work
      // properly in some browsers, e.g.
      // https://bugzilla.mozilla.org/show_bug.cgi?id=1335594
      // tslint:disable-next-line: no-any
      imageBitmap = (await createImageBitmap(
              pixels, CreateImageBitmapOptions(premultiplyAlpha: 'none')))
          as ImageBitmap;
    } catch (e) {
      imageBitmap = null;
    }

    // createImageBitmap will clip the source size.
    // In some cases, the input will have larger size than its content.
    // E.g. new Image(10, 10) but with 1 x 1 content. Using
    // createImageBitmap will clip the size from 10 x 10 to 1 x 1, which
    // is not correct. We should avoid wrapping such resouce to
    // imageBitmap.
    if (imageBitmap != null &&
        imageBitmap.width == (pixels as dynamic).width &&
        imageBitmap.height == (pixels as dynamic).height) {
      inputs = imageBitmap;
    } else {
      inputs = pixels;
    }
  } else {
    inputs = pixels;
  }

  return fromPixels(inputs, numChannels: numChannels);
}

/**
 * Draws a `tf.Tensor` of pixel values to a byte array or optionally a
 * canvas.
 *
 * When the dtype of the input is 'float32', we assume values in the range
 * [0-1]. Otherwise, when input is 'int32', we assume values in the range
 * [0-255].
 *
 * Returns a promise that resolves when the canvas has been drawn to.
 *
 * @param img A rank-2 tensor with shape `[height, width]`, or a rank-3 tensor
 * of shape `[height, width, numChannels]`. If rank-2, draws grayscale. If
 * rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
 * grayscale. When depth of 3, we draw with the first three components of
 * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
 * 4, all four components of the depth dimension correspond to r, g, b, a.
 * @param canvas The canvas to draw to.
 *
 * @doc {heading: 'Browser', namespace: 'browser'}
 */
Future<Uint8ClampedList> toPixels(
  Tensor img, //: Tensor2D|Tensor3D|TensorLike,
  CanvasElement? canvas,
) async {
  var $img = convertToTensor(img, 'img', 'toPixels');
  if (img is! Tensor) {
    // Assume int32 if user passed a native array.
    final originalImgTensor = $img;
    $img = cast(originalImgTensor, 'int32');
    originalImgTensor.dispose();
  }
  if ($img.rank != 2 && $img.rank != 3) {
    throw Exception(
        "toPixels only supports rank 2 or 3 tensors, got rank ${$img.rank}.");
  }
  final height = $img.shape[0];
  final width = $img.shape[1];
  final depth = $img.rank == 2 ? 1 : $img.shape[2];

  if (depth > 4 || depth == 2) {
    throw Exception(
        "toPixels only supports depth of size " + "1, 3 or 4 but got ${depth}");
  }

  if ($img.dtype != 'float32' && $img.dtype != 'int32') {
    throw Exception("Unsupported type for toPixels: ${$img.dtype}." +
        " Please use float32 or int32 tensors.");
  }

  final data = await $img.data();
  final multiplier = $img.dtype == 'float32' ? 255 : 1;
  final bytes = Uint8ClampedList(width * height * 4);

  for (int i = 0; i < height * width; ++i) {
    final rgba = [0, 0, 0, 255];

    for (int d = 0; d < depth; d++) {
      final value = data[i * depth + d];

      if ($img.dtype == 'float32') {
        if (value < 0 || value > 1) {
          throw Exception("Tensor values for a float32 Tensor must be in the " +
              "range [0 - 1] but encountered ${value}.");
        }
      } else if ($img.dtype == 'int32') {
        if (value < 0 || value > 255) {
          throw Exception("Tensor values for a int32 Tensor must be in the " +
              "range [0 - 255] but encountered ${value}.");
        }
      }

      if (depth == 1) {
        rgba[0] = value * multiplier;
        rgba[1] = value * multiplier;
        rgba[2] = value * multiplier;
      } else {
        rgba[d] = value * multiplier;
      }
    }

    final j = i * 4;
    bytes[j + 0] = (rgba[0]).round();
    bytes[j + 1] = (rgba[1]).round();
    bytes[j + 2] = (rgba[2]).round();
    bytes[j + 3] = (rgba[3]).round();
  }

  if (canvas != null) {
    canvas.width = width;
    canvas.height = height;
    final ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    final imageData = ImageData(bytes, width, height);
    ctx.putImageData(imageData, 0, 0);
  }
  if ($img != img) {
    $img.dispose();
  }
  return bytes;
}
