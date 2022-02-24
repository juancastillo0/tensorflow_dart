import 'non_max_suppression.dart' as non_max_suppression;
import 'non_max_suppression_async.dart' as non_max_suppression_async;
import 'non_max_suppression_padded.dart' as non_max_suppression_padded;
import 'non_max_suppression_with_score.dart' as non_max_suppression_with_score;

import 'crop_and_resize.dart' as crop_and_resize;
import 'resize_bilinear.dart' as resize_bilinear;
import 'resize_nearest_neighbor.dart' as resize_nearest_neighbor;
import 'transform.dart' as transform_;

// ignore: camel_case_types
class image {
  const image._();

  static const cropAndResize = crop_and_resize.cropAndResize;
  static const resizeBilinear = resize_bilinear.resizeBilinear;
  static const resizeNearestNeighbor =
      resize_nearest_neighbor.resizeNearestNeighbor;
  static const transform = transform_.transform;

  static const nonMaxSuppression = non_max_suppression.nonMaxSuppression;
  static const nonMaxSuppressionAsync =
      non_max_suppression_async.nonMaxSuppressionAsync;
  static const nonMaxSuppressionPadded =
      non_max_suppression_padded.nonMaxSuppressionPadded;
  static const nonMaxSuppressionWithScore =
      non_max_suppression_with_score.nonMaxSuppressionWithScore;
}
