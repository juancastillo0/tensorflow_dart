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

// import {AnchorConfig} from './interfaces/config_interfaces';
// import {Rect} from './interfaces/shape_interfaces';

import 'dart:math' as Math;

import 'interfaces/config_interfaces.dart';
import 'interfaces/shape_interfaces.dart';

// ref:
// https://github.com/google/mediapipe/blob/350fbb2100ad531bc110b93aaea23d96af5a5064/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
List<Rect> createSsdAnchors(AnchorConfig config) {
  // Set defaults.
  final reduceBoxesInLowestLayer = config.reduceBoxesInLowestLayer ?? false;
  final interpolatedScaleAspectRatio =
      config.interpolatedScaleAspectRatio ?? 1.0;
  final fixedAnchorSize = config.fixedAnchorSize ?? false;

  final List<Rect> anchors = [];
  int layerId = 0;
  while (layerId < config.numLayers) {
    final List<double> anchorHeight = [];
    final List<double> anchorWidth = [];
    final List<double> aspectRatios = [];
    final List<double> scales = [];

    // For same strides, we merge the anchors in the same order.
    int lastSameStrideLayer = layerId;
    while (lastSameStrideLayer < config.strides.length &&
        config.strides[lastSameStrideLayer] == config.strides[layerId]) {
      final scale = _calculateScale(config.minScale, config.maxScale,
          lastSameStrideLayer, config.strides.length);
      if (lastSameStrideLayer == 0 && reduceBoxesInLowestLayer) {
        // For first layer, it can be specified to use predefined anchors.
        aspectRatios.add(1);
        aspectRatios.add(2);
        aspectRatios.add(0.5);
        scales.add(0.1);
        scales.add(scale);
        scales.add(scale);
      } else {
        for (int aspectRatioId = 0;
            aspectRatioId < config.aspectRatios.length;
            ++aspectRatioId) {
          aspectRatios.add(config.aspectRatios[aspectRatioId]);
          scales.add(scale);
        }
        if (interpolatedScaleAspectRatio > 0.0) {
          final scaleNext = lastSameStrideLayer == config.strides.length - 1
              ? 1.0
              : _calculateScale(config.minScale, config.maxScale,
                  lastSameStrideLayer + 1, config.strides.length);
          scales.add(Math.sqrt(scale * scaleNext));
          aspectRatios.add(interpolatedScaleAspectRatio);
        }
      }
      lastSameStrideLayer++;
    }

    for (int i = 0; i < aspectRatios.length; ++i) {
      final ratioSqrts = Math.sqrt(aspectRatios[i]);
      anchorHeight.add(scales[i] / ratioSqrts);
      anchorWidth.add(scales[i] * ratioSqrts);
    }

    int featureMapHeight = 0;
    int featureMapWidth = 0;
    if (config.featureMapHeight.length > 0) {
      featureMapHeight = config.featureMapHeight[layerId];
      featureMapWidth = config.featureMapWidth[layerId];
    } else {
      final stride = config.strides[layerId];
      featureMapHeight = (config.inputSizeHeight / stride).ceil();
      featureMapWidth = (config.inputSizeWidth / stride).ceil();
    }

    for (int y = 0; y < featureMapHeight; ++y) {
      for (int x = 0; x < featureMapWidth; ++x) {
        for (int anchorId = 0; anchorId < anchorHeight.length; ++anchorId) {
          final xCenter = (x + config.anchorOffsetX) / featureMapWidth;
          final yCenter = (y + config.anchorOffsetY) / featureMapHeight;

          final double width;
          final double height;
          if (fixedAnchorSize) {
            width = 1.0;
            height = 1.0;
          } else {
            width = anchorWidth[anchorId];
            height = anchorHeight[anchorId];
          }
          final newAnchor = Rect(
            xCenter: xCenter,
            yCenter: yCenter,
            width: width,
            height: height,
          );

          anchors.add(newAnchor);
        }
      }
    }
    layerId = lastSameStrideLayer;
  }

  return anchors;
}

double _calculateScale(
  double minScale,
  double maxScale,
  int strideIndex,
  int numStrides,
) {
  if (numStrides == 1) {
    return (minScale + maxScale) * 0.5;
  } else {
    return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1);
  }
}
