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

// import {Keypoint} from './common_interfaces';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'common_interfaces.dart';

/**
 * A rectangle that contains center point, height, width and rotation info.
 * Can be normalized or non-normalized.
 */
class Rect implements ImageSize {
  final double xCenter;
  final double yCenter;
  final double height;
  final double width;
  final double rotation;

  const Rect({
    required this.xCenter,
    required this.yCenter,
    required this.height,
    required this.width,
    this.rotation = 0,
  });

  Rect copyWith({
    double? xCenter,
    double? yCenter,
    double? height,
    double? width,
    double? rotation,
  }) {
    return Rect(
      xCenter: xCenter ?? this.xCenter,
      yCenter: yCenter ?? this.yCenter,
      height: height ?? this.height,
      width: width ?? this.width,
      rotation: rotation ?? this.rotation,
    );
  }
}

class BoundingBox {
  final double xMin;
  final double yMin;
  final double xMax;
  final double yMax;
  final double width;
  final double height;

  const BoundingBox({
    required this.xMin,
    required this.yMin,
    required this.xMax,
    required this.yMax,
    required this.width,
    required this.height,
  });
}

class AnchorTensor {
  final tf.Tensor1D x;
  final tf.Tensor1D y;
  final tf.Tensor1D w;
  final tf.Tensor1D h;

  const AnchorTensor({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
  });
}

class LocationData {
  final BoundingBox? boundingBox;
  final BoundingBox relativeBoundingBox;
  final List<Keypoint> relativeKeypoints;

  LocationData({
    this.boundingBox,
    required this.relativeBoundingBox,
    List<Keypoint>? relativeKeypoints,
  }) : relativeKeypoints = relativeKeypoints ?? [];
}

class Detection {
  final List<double>? score;
  // Location data corresponding to all
  // detected labels above.
  final LocationData locationData;
  // i-th label or labelid has a score encoded by the i-th
  // element in score. Either string or integer labels must
  // be used but not both.
  final List<String>? label;
  final List<int>? labelId;
  final int? ind;

  const Detection({
    this.score,
    required this.locationData,
    this.label,
    this.labelId,
    this.ind,
  });

  Detection copyWith({
    List<double>? score,
    LocationData? locationData,
    List<String>? label,
    List<int>? labelId,
    int? ind,
  }) {
    return Detection(
      score: score ?? this.score,
      locationData: locationData ?? this.locationData,
      label: label ?? this.label,
      labelId: labelId ?? this.labelId,
      ind: ind ?? this.ind,
    );
  }
}
