import 'package:tensorflow_wasm/src/kernel_registry.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/add_n.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/all.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/any.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/arg_max.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/avg_pool.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/base_kernels.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/batch_mat_mul.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/batch_to_space_nd.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/clip_by_value.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/concat.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/crop_and_resize.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/cumsum.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/depth_to_space.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/depthwise_conv2d_native.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/expand_dims.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/fill.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/fused_batch_norm.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/fused_conv2d.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/fused_depthwise_conv2d.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/fused_mat_mul_.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/gather_nd.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/gather_v2.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/identity.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/leaky_relu.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/max.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/max_pool.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/mean.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/mirror_pad.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/non_max_suppression_v3.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/non_max_suppression_v4.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/non_max_suppression_v5.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/one_hot.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/ones_like.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/pack.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/pad_v2.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/prelu.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/range.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/reshape.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/resize_bilinear.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/reverse.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/rotate_with_offset.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/scatter_nd.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/select.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/sigmoid.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/slice.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/softmax.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/space_to_batch_nd.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/sparse_fill_empty_rows.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/sparse_reshape.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/sparse_segment_mean.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/sparse_segment_sum.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/split_v.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/step.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/sum.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/tile.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/top_k.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/transform.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/transpose.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/unpack.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/zeros_like.dart';

void registerAllKernels() {
  final configs = <KernelConfig>[
    addConfig,
    realDivConfig,
    squaredDifferenceConfig,
    greaterConfig,
    lessConfig,
    maximumConfig,
    minimumConfig,
    floorDivConfig,
    multiplyConfig,
    powConfig,
    subConfig,
    equalConfig,
    greaterEqualConfig,
    lessEqualConfig,
    logicalAndConfig,
    notEqualConfig,
    expConfig,
    eluConfig,
    coshConfig,
    tanConfig,
    ceilConfig,
    roundConfig,
    floorConfig,
    negConfig,
    rsqrtConfig,
    relu6Config,
    squareConfig,
    tanhConfig,
    logConfig,
    cosConfig,
    sinConfig,
    absConfig,
    sqrtConfig,
    reluConfig,
    //
    identityConfig,
    addNConfig,
    batchMatMulConfig,
    argMaxConfig,
    allConfig,
    anyConfig,
    minConfig,
    maxConfig,
    sumConfig,
    prodConfig,
    cumsumConfig,
    //
    transposeConfig,
    concatConfig,
    expandDimsConfig,
    packConfig,
    reshapeConfig,
    selectConfig,
    tileConfig,
    fusedBatchNormConfig,
    fusedConv2DConfig,
    fusedDepthwiseConv2DConfig,
    fusedMatMulConfig_,
    gatherNdConfig,
    gatherV2Config,
    scatterNdConfig,
    sigmoidConfig,
    sliceConfig,
    softmaxConfig,
    splitVConfig,
    topKConfig,
    unpackConfig,
    rangeConfig,
    zerosLikeConfig,
    onesLikeConfig,
    oneHotConfig,
    fillConfig,
    batchToSpaceNDConfig,
    depthToSpaceConfig,
    mirrorPadConfig,
    padV2Config,
    spaceToBatchNDConfig,
    cropAndResizeConfig,
    nonMaxSuppressionV3Config,
    nonMaxSuppressionV4Config,
    nonMaxSuppressionV5Config,
    resizeBilinearConfig,
    rotateWithOffsetConfig,
    clipByValueConfig,
    transformConfig,
    //
    depthwiseConv2dNativeConfig,
    leakyReluConfig,
    preluConfig,
    reverseConfig,
    stepConfig,
    meanConfig,
    avgPoolConfig,
    maxPoolConfig,

    // Conv2D
    // Conv2DBackpropInput
    sparseFillEmptyRowsConfig,
    sparseReshapeConfig,
    sparseSegmentMeanConfig,
    sparseSegmentSumConfig,
  ];

  for (final config in configs) {
    registerKernel(config);
  }
}
