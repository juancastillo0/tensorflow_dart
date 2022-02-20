import 'package:tensorflow_wasm/src/kernel_registry.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/add_n.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/all.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/any.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/arg_max.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/base_kernels.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/batch_mat_mul.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/concat.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/cumsum.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/depthwise_conv2d_native.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/expand_dims.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/identity.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/max.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/pack.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/reshape.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/select.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/sum.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/tile.dart';
import 'package:tensorflow_wasm/src/wasm/kernels/transpose.dart';

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
    //
    depthwiseConv2dNativeConfig,
    // Mean
    // FusedBatchNorm
    // FusedConv2D
    // FusedDepthwiseConv2D
    // OneHot
    // Sigmoid
    // Softmax
    // TopK
    // _FusedMatMul

    // AvgPool
    // BatchToSpaceND
    // ClipByValue
    // Conv2D
    // Conv2DBackpropInput
    // CropAndResize
    // DepthToSpace
    // Fill
    // FlipLeftRight
    // FusedBatchNorm
    // FusedConv2D
    // FusedDepthwiseConv2D
    // GatherNd
    // GatherV2
    // LeakyRelu
    // MaxPool
    // MirrorPad
    // NonMaxSuppressionV3.ts
    // NonMaxSuppressionV4.ts
    // NonMaxSuppressionV5.ts
    // NonMaxSuppression_util.ts
    // OneHot
    // OnesLike
    // PadV2
    // Prelu
    // Range
    // ResizeBilinear
    // Reverse
    // RotateWithOffset
    // ScatterNd
    // Slice
    // SpaceToBatchND
    // SparseFillEmptyRows
    // SparseReshape
    // SparseSegmentMean
    // SparseSegmentReduction
    // SparseSegmentSum
    // SplitV
    // Step
    // StridedSlice
    // Transform
    // Unpack
    // ZerosLike
  ];

  for (final config in configs) {
    registerKernel(config);
  }
}
