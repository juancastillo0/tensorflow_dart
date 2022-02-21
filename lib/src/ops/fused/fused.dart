import 'package:tensorflow_wasm/src/ops/fused/conv2d.dart';
import 'package:tensorflow_wasm/src/ops/fused/depthwise_conv2d.dart';
import 'package:tensorflow_wasm/src/ops/fused/mat_mul.dart';

const fused = FusedOps._();

class FusedOps {
  const FusedOps._();

  final conv2d = fusedConv2d;
  final matMul = fusedMatMul;
  final depthwiseConv2d = fusedDepthwiseConv2d;
}