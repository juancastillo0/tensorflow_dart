import 'sparse_fill_empty_rows.dart' as sparse_fill_empty_rows;
import 'sparse_reshape.dart' as sparse_reshape;
import 'sparse_segment_mean.dart' as sparse_segment_mean;
import 'sparse_segment_sum.dart' as sparse_segment_sum;

// ignore: camel_case_types
class sparse {
  static const sparseFillEmptyRows = sparse_fill_empty_rows.sparseFillEmptyRows;
  static const sparseReshape = sparse_reshape.sparseReshape;
  static const sparseSegmentMean = sparse_segment_mean.sparseSegmentMean;
  static const sparseSegmentSum = sparse_segment_sum.sparseSegmentSum;
}
