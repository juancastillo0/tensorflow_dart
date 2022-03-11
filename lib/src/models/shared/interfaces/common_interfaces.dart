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

// import {Tensor3D} from '@tensorflow/tfjs-core';
import 'package:universal_html/html.dart' as html;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' show Tensor3D;

class Nullable<T extends Object> {
  final T? value;

  const Nullable(this.value);
}

typedef PixelInput = Object;
//  Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
//     HTMLCanvasElement|ImageBitmap;

class InputResolution implements ImageSize {
  final double width;
  final double height;

  const InputResolution({
    required this.width,
    required this.height,
  });
}

/**
 * A keypoint that contains coordinate information.
 */
class Keypoint {
  final double x;
  final double y;
  final double? z;
  final double? score; // The probability of a keypoint's visibility.
  final String? name;

  const Keypoint({
    required this.x,
    required this.y,
    this.z,
    this.score,
    this.name,
  });

  Keypoint copyWith({
    double? x,
    double? y,
    Nullable<double>? z,
    Nullable<double>? score,
    Nullable<String>? name,
  }) {
    return Keypoint(
      x: x ?? this.x,
      y: y ?? this.y,
      z: z != null ? z.value : this.z,
      score: score != null ? score.value : this.score,
      name: name != null ? name.value : this.name,
    );
  }
}

class ImageSize {
  final double height;
  final double width;

  ImageSize({
    required this.height,
    required this.width,
  });
}

class Padding {
  final int top;
  final int bottom;
  final int left;
  final int right;

  Padding({
    required this.top,
    required this.bottom,
    required this.left,
    required this.right,
  });
}

class ValueTransform {
  final double scale;
  final double offset;

  ValueTransform({
    required this.scale,
    required this.offset,
  });
}

class WindowElement {
  final double distance;
  final double duration;

  WindowElement({
    required this.distance,
    required this.duration,
  });
}

abstract class KeypointsFilter {
  List<Keypoint> apply(
    List<Keypoint> landmarks,
    double microSeconds,
    double objectScale,
  );
  void reset();
}

abstract class Mask {
  Future<html.CanvasImageSource>
      toCanvasImageSource(); /* RGBA image of same size as input, where
                            mask semantics are green and blue are always set to
                            0. Different red values denote different body
                            parts(see maskValueToBodyPart explanation below).
                            Different alpha values denote the probability of
                            pixel being a foreground pixel (0 being lowest
                            probability and 255 being highest).*/

  Future<html.ImageData>
      toImageData(); /* 1 dimensional array of size image width * height *
                    4, where each pixel is represented by RGBA in that order.
                    For each pixel, the semantics are green and blue are always
                    set to 0, and different red values denote different body
                    parts (see maskValueToBodyPart explanation below). Different
                    alpha values denote the probability of the pixel being a
                    foreground pixel (0 being lowest probability and 255 being
                    highest). */

  Future<Tensor3D>
      toTensor(); /* RGBA image of same size as input, where mask
                   semantics are green and blue are always set to 0. Different
                   red values denote different body parts (see
                   maskValueToBodyPart explanation below). Different alpha
                   values denote the probability of pixel being a foreground
                   pixel (0 being lowest probability and 255 being highest).*/

  // 'canvasimagesource'|'imagedata'| 'tensor'
  String
      getUnderlyingType(); /* determines which type the mask currently stores in its
                   implementation so that conversion can be avoided */
}

abstract class Segmentation {
  String maskValueToLabel(
      double
          maskValue); /* Maps a foreground pixel’s red value to the segmented part name
                 of that pixel. Should throw error for unsupported input
                 values.*/
  Mask get mask;
}

class Color {
  final int r;
  final int g;
  final int b;
  final int a;

  const Color({
    required this.r,
    required this.g,
    required this.b,
    required this.a,
  });
}
