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

// import {InputResolution} from './common_interfaces';
import 'common_interfaces.dart' show InputResolution;

class ImageToTensorConfig {
  final InputResolution outputTensorSize;
  final bool? keepAspectRatio;
  final List<double>? outputTensorFloatRange; // [number, number]
  final BorderMode borderMode;

  const ImageToTensorConfig({
    required this.outputTensorSize,
    this.keepAspectRatio,
    this.outputTensorFloatRange,
    required this.borderMode,
  });
}

enum BorderMode {
  zero,
  replicate,
}

class VelocityFilterConfig {
  final int? windowSize; // Number of value changes to keep over time. Higher
  // value adds to lag and to stability.
  final int?
      velocityScale; // Scale to apply to the velocity calculated over the given
  // window. With higher velocity `low pass filter` weights new new
  // values higher. Lower value adds to lag and to stability.
  final int?
      minAllowedObjectScale; // If calculated object scale is less than given value smoothing
  // will be disabled and landmarks will be returned as is.
  final bool?
      disableValueScaling; // Disable value scaling based on object size and use `1.0`
  // instead. Value scale is calculated as inverse value of object
  // size. Object size is calculated as maximum side of
  // rectangular bounding box of the object in XY plane.

  const VelocityFilterConfig({
    this.windowSize,
    this.velocityScale,
    this.minAllowedObjectScale,
    this.disableValueScaling,
  });
}

class OneEuroFilterConfig {
  final int? frequency; // Frequency of incoming frames defined in seconds. Used
  // only if can't be calculated from provided events (e.g.
  // on the very fist frame).
  final int? minCutOff;
  // Minimum cutoff frequency. Start by tuning this parameter while
  // keeping `beta=0` to reduce jittering to the desired level. 1Hz
  // (the default value) is a a good starting point.
  final int? beta; // Cutoff slope. After `minCutOff` is configured, start
  // increasing `beta` value to reduce the lag introduced by the
  // `minCutoff`. Find the desired balance between jittering and
  // lag.
  final int? derivateCutOff;
  // Cutoff frequency for derivate. It is set to 1Hz in the
  // original algorithm, but can be turned to further smooth the
  // speed (i.e. derivate) on the object.
  final int?
      thresholdCutOff; // The outlier threshold offset, will lead to a generally more
  // reactive filter that will be less likely to discount outliers.
  final int?
      thresholdBeta; // The outlier threshold slope, will lead to a filter
  // that will more aggressively react whenever the
  // keypoint speed increases by being less likely to
  // consider that an observation is an outlier.
  final int?
      minAllowedObjectScale; // If calculated object scale is less than given value smoothing
  // will be disabled and keypoints will be returned as is. This
  // value helps filter adjust to the distance to camera.
  final bool?
      disableValueScaling; // Disable value scaling based on object size and use `1.0`
  // instead. Value scale is calculated as inverse value of object
  // size. Object size is calculated as maximum side of
  // rectangular bounding box of the object in XY plane.

  const OneEuroFilterConfig({
    this.frequency,
    this.minCutOff,
    this.beta,
    this.derivateCutOff,
    this.thresholdCutOff,
    this.thresholdBeta,
    this.minAllowedObjectScale,
    this.disableValueScaling,
  });
}

class KeypointsSmoothingConfig {
  final VelocityFilterConfig? velocityFilter;
  final OneEuroFilterConfig? oneEuroFilter;

  const KeypointsSmoothingConfig({
    this.velocityFilter,
    this.oneEuroFilter,
  });
}

class AnchorConfig {
  final int
      numLayers; // Number of output feature maps to generate the anchors on.
  final double
      minScale; // Min scales for generating anchor boxes on feature maps.
  final double
      maxScale; // Max scales for generating anchor boxes on feature maps.
  final int inputSizeHeight; // Size of input images.
  final int inputSizeWidth;
  final double
      anchorOffsetX; // The offset for the center of anchors. The values is
  // in the scale of stride. E.g. 0.5 meaning 0.5 *
  // currentStride in pixels.
  final double anchorOffsetY;
  final List<int>
      featureMapWidth; // Sizes of output feature maps to create anchors. Either
  // featureMap size or stride should be provided.
  final List<int> featureMapHeight;
  final List<int> strides; // Strides of each output feature maps.
  final List<double>
      aspectRatios; // List of different aspect ratio to generate anchors.
  final bool?
      fixedAnchorSize; // Whether use fixed width and height (e.g. both 1.0) for each
  // anchor. This option can be used when the predicted anchor
  // width and height are in pixels.
  final bool?
      reduceBoxesInLowestLayer; // A boolean to indicate whether the fixed 3 boxes per location
  // is used in the lowest layer.
  final double?
      interpolatedScaleAspectRatio; // An additional anchor is added with this aspect ratio and a
  // scale interpolated between the scale for a layer and the scale
  // for the next layer (1.0 for the last layer). This anchor is
  // not included if this value is 0.

  const AnchorConfig({
    required this.numLayers,
    required this.minScale,
    required this.maxScale,
    required this.inputSizeHeight,
    required this.inputSizeWidth,
    required this.anchorOffsetX,
    required this.anchorOffsetY,
    required this.featureMapWidth,
    required this.featureMapHeight,
    required this.strides,
    required this.aspectRatios,
    this.fixedAnchorSize,
    this.reduceBoxesInLowestLayer,
    this.interpolatedScaleAspectRatio,
  });
}

class TensorsToDetectionsConfig {
  final int
      numClasses; // The number of output classes predicted by the detection model.
  final int
      numBoxes; // The number of output boxes predicted by the detection model.
  final int numCoords; // The number of output values per boxes predicted by the
  // detection model. The values contain bounding boxes,
  // keypoints, etc.
  final int? keypointCoordOffset; // The offset of keypoint coordinates in the
  // location tensor.
  final int? numKeypoints; // The number of predicted keypoints.
  final int?
      numValuesPerKeypoint; // The dimension of each keypoint, e.g. number
  // of values predicted for each keypoint.
  final int? boxCoordOffset; // The offset of box coordinates in the location
  // tensor.
  final double? xScale; // Parameters for decoding SSD detection model.
  final double? yScale; // Parameters for decoding SSD detection model.
  final double? wScale; // Parameters for decoding SSD detection model.
  final double? hScale; // Parameters for decoding SSD detection model.
  final bool? applyExponentialOnBoxSize;
  final bool?
      reverseOutputOrder; // Whether to reverse the order of predicted x,
  // y from output. If false, the order is
  // [y_center, x_center, h, w], if true the
  // order is [x_center, y_center, w, h].
  final List<int>?
      ignoreClasses; // The ids of classes that should be ignored during
  // decoding the score for each predicted box.
  final bool? sigmoidScore;
  final double? scoreClippingThresh;
  final bool?
      flipVertically; // Whether the detection coordinates from the input
  // tensors should be flipped vertically (along the
  // y-direction).
  final double?
      minScoreThresh; // Score threshold for perserving decoded detections.

  const TensorsToDetectionsConfig({
    required this.numClasses,
    required this.numBoxes,
    required this.numCoords,
    this.keypointCoordOffset,
    this.numKeypoints,
    this.numValuesPerKeypoint,
    this.boxCoordOffset,
    this.xScale,
    this.yScale,
    this.wScale,
    this.hScale,
    this.applyExponentialOnBoxSize,
    this.reverseOutputOrder,
    this.ignoreClasses,
    this.sigmoidScore,
    this.scoreClippingThresh,
    this.flipVertically,
    this.minScoreThresh,
  });
}

class DetectionToRectConfig {
  final int rotationVectorStartKeypointIndex;
  final int rotationVectorEndKeypointIndex;
  final double? rotationVectorTargetAngleDegree; // In degrees.
  final double? rotationVectorTargetAngle; // In radians.

  DetectionToRectConfig({
    required this.rotationVectorStartKeypointIndex,
    required this.rotationVectorEndKeypointIndex,
    this.rotationVectorTargetAngleDegree,
    this.rotationVectorTargetAngle,
  });
}

class RectTransformationConfig {
  final double scaleX; // Scaling factor along the side of a rotated rect that
  // was aligned with the X and Y axis before rotation
  // respectively.
  final double scaleY;
  final double? rotation; // Additional rotation (counter-clockwise) around the
  // rect center either in radians or in degrees.
  final double? rotationDegree;
  final double shiftX; // Shift along the side of a rotated rect that was
  // aligned with the X and Y axis before rotation
  // respectively. The shift is relative to the length of
  // corresponding side. For example, for a rect with size
  // (0.4, 0.6), with shiftX = 0.5 and shiftY = -0.5 the
  // rect is shifted along the two sides by 0.2 and -0.3
  // respectively.
  final double shiftY;
  final bool? squareLong; // Change the final transformed rect into a square
  // that shares the same center and rotation with
  // the rect, and with the side of the square equal
  // to either the long or short side of the rect
  // respectively.
  final bool? squareShort;

  const RectTransformationConfig({
    required this.scaleX,
    required this.scaleY,
    this.rotation,
    this.rotationDegree,
    required this.shiftX,
    required this.shiftY,
    this.squareLong,
    this.squareShort,
  });
}

class TensorsToLandmarksConfig {
  final int numLandmarks;
  final int inputImageWidth;
  final int inputImageHeight;
  final String visibilityActivation; // 'none'|'sigmoid'
  final bool? flipHorizontally;
  final bool? flipVertically;
  final double? normalizeZ;

  const TensorsToLandmarksConfig({
    required this.numLandmarks,
    required this.inputImageWidth,
    required this.inputImageHeight,
    required this.visibilityActivation,
    this.flipHorizontally,
    this.flipVertically,
    this.normalizeZ,
  });
}

class TensorsToSegmentationConfig {
  final String activation; //: 'none'|'sigmoid'|'softmax';

  const TensorsToSegmentationConfig({
    required this.activation,
  });
}

class SegmentationSmoothingConfig {
  final double combineWithPreviousRatio;

  const SegmentationSmoothingConfig({
    required this.combineWithPreviousRatio,
  });
}

class RefineLandmarksFromHeatmapConfig {
  final int? kernelSize;
  final double minConfidenceToRefine;

  const RefineLandmarksFromHeatmapConfig({
    this.kernelSize,
    required this.minConfidenceToRefine,
  });
}

class VisibilitySmoothingConfig {
  final double alpha; // Coefficient applied to a new value, and `1 - alpha` is
  // applied to a stored value. Should be in [0, 1] range. The
  // smaller the value, the smoother result and the bigger lag.

  const VisibilitySmoothingConfig({
    required this.alpha,
  });
}

typedef AssignAverage = List<int>;

/// 'none'|'copy'|AssignAverage;  // Z refinement instructions.
typedef LandmarksRefinementConfigZRefinement = Object;

class LandmarksRefinementConfig {
  final List<int>
      indexesMapping; // Maps indexes of the given set of landmarks to indexes of the
  // resulting set of landmarks. Should be non empty and contain
  // the same amount of indexes as landmarks in the corresponding
  // input.
  final LandmarksRefinementConfigZRefinement
      zRefinement; //: 'none'|'copy'|AssignAverage;  // Z refinement instructions.

  const LandmarksRefinementConfig({
    required this.zRefinement,
    required this.indexesMapping,
  });
}
