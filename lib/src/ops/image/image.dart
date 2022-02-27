import '../../tensor.dart';
import 'non_max_suppression.dart' as non_max_suppression;
import 'non_max_suppression_async.dart' as non_max_suppression_async;
import 'non_max_suppression_padded.dart' as non_max_suppression_padded;
import 'non_max_suppression_with_score.dart' as non_max_suppression_with_score;
import 'non_max_suppression_padded_async.dart'
    as non_max_suppression_padded_async;
import 'non_max_suppression_with_score_async.dart'
    as non_max_suppression_with_score_async;

import 'crop_and_resize.dart' as crop_and_resize;
import 'flip_left_right.dart' as flip_left_right;
import 'grayscale_to_rgb.dart' as grayscale_to_rgb;
import 'rotate_with_offset.dart' as rotate_with_offset;
import 'resize_bilinear.dart' as resize_bilinear;
import 'resize_nearest_neighbor.dart' as resize_nearest_neighbor;
import 'transform.dart' as transform_;

// ignore: camel_case_types
class image {
  const image._();

  static const cropAndResize = crop_and_resize.cropAndResize;
  static const resizeBilinear = resize_bilinear.resizeBilinear;
  static const flipLeftRight = flip_left_right.flipLeftRight;
  static const rotateWithOffset = rotate_with_offset.rotateWithOffset;
  static const grayscaleToRGB = grayscale_to_rgb.grayscaleToRGB;
  static const resizeNearestNeighbor =
      resize_nearest_neighbor.resizeNearestNeighbor;
  static const transform = transform_.transform;

  static const nonMaxSuppression = non_max_suppression.nonMaxSuppression;
  static const nonMaxSuppressionWithScoreAsync =
      non_max_suppression_with_score_async.nonMaxSuppressionWithScoreAsync;
  static const nonMaxSuppressionPaddedAsync =
      non_max_suppression_padded_async.nonMaxSuppressionPaddedAsync;
  static const nonMaxSuppressionAsync =
      non_max_suppression_async.nonMaxSuppressionAsync;
  static const nonMaxSuppressionPadded =
      non_max_suppression_padded.nonMaxSuppressionPadded;
  static const nonMaxSuppressionWithScore =
      non_max_suppression_with_score.nonMaxSuppressionWithScore;

  static const double defaultIouThreshold = 0.5;
  static const double defaultScoreThreshold = double.negativeInfinity;
  static const double defaultSoftNmsSigma = 0.0;
}

class NmsPadded {
  final Tensor1D selectedIndices;
  final Scalar validOutputs;

  NmsPadded({
    required this.selectedIndices,
    required this.validOutputs,
  });
}

class NmsWithScore {
  final Tensor1D selectedIndices;
  final Tensor1D selectedScores;

  NmsWithScore({
    required this.selectedIndices,
    required this.selectedScores,
  });
}
