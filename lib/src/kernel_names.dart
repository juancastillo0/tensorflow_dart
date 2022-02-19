// ignore_for_file: constant_identifier_names

import 'dart:collection';

import 'package:tensorflow_wasm/src/kernel_registry.dart';
import 'package:tensorflow_wasm/src/tensor.dart';

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Allow UpperCamelCase variable names
// tslint:disable: variable-name
// Unfortunately just enabling PascalCase per file (tslint:enable:
// allow-pascal-case) doesn't work.

// commit: 38367e3385f640e36108a61f8e98f854d6237a90
// import {NamedTensorInfoMap, TensorInfo} from './kernel_registry';
// import {ExplicitPadding} from './ops/conv_util';
// import {Activation} from './ops/fused_types';
// import {DataType, PixelData} from './types';

const Abs = 'Abs';
typedef AbsInputs = UnaryInputs;

// export const Acos = 'Acos';
// export type AcosInputs = UnaryInputs;

// export const Acosh = 'Acosh';
// export type AcoshInputs = UnaryInputs;

const Add = 'Add';
typedef AddInputs = BinaryInputs;

// export const AddN = 'AddN';
// export type AddNInputs = TensorInfo[];

// export const All = 'All';
// export type AllInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface AllAttrs {
//   axis: number|number[];
//   keepDims: boolean;
// }

// export const Any = 'Any';
// export type AnyInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface AnyAttrs {
//   axis: number|number[];
//   keepDims: boolean;
// }

// export const ArgMax = 'ArgMax';
// export type ArgMaxInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface ArgMaxAttrs {
//   axis: number;
// }

// export const ArgMin = 'ArgMin';
// export type ArgMinInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface ArgMinAttrs {
//   axis: number;
// }

// export const Asin = 'Asin';
// export type AsinInputs = UnaryInputs;

// export const Asinh = 'Asinh';
// export type AsinhInputs = UnaryInputs;

// export const Atan = 'Atan';
// export type AtanInputs = UnaryInputs;

// export const Atanh = 'Atanh';
// export type AtanhInputs = UnaryInputs;

// export const Atan2 = 'Atan2';
// export type Atan2Inputs = BinaryInputs;

// export const AvgPool = 'AvgPool';
// export type AvgPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface AvgPoolAttrs {
//   filterSize: [number, number]|number;
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
// }

// export const AvgPoolGrad = 'AvgPoolGrad';
// export type AvgPoolGradInputs = Pick<NamedTensorInfoMap, 'dy'|'input'>;
// export interface AvgPoolGradAttrs {
//   filterSize: [number, number]|number;
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
// }

// export const AvgPool3D = 'AvgPool3D';
// export type AvgPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface AvgPool3DAttrs {
//   filterSize: [number, number, number]|number;
//   strides: [number, number, number]|number;
//   pad: 'valid'|'same'|number;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
//   dataFormat: 'NDHWC'|'NCDHW';
// }

// export const AvgPool3DGrad = 'AvgPool3DGrad';
// export type AvgPool3DGradInputs = Pick<NamedTensorInfoMap, 'dy'|'input'>;
// export interface AvgPool3DGradAttrs {
//   filterSize: [number, number, number]|number;
//   strides: [number, number, number]|number;
//   pad: 'valid'|'same'|number;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
// }

// export const BatchMatMul = 'BatchMatMul';
// export type BatchMatMulInputs = Pick<NamedTensorInfoMap, 'a'|'b'>;
// export interface BatchMatMulAttrs {
//   transposeA: boolean;
//   transposeB: boolean;
// }

// export const BatchToSpaceND = 'BatchToSpaceND';
// export type BatchToSpaceNDInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface BatchToSpaceNDAttrs {
//   blockShape: number[];
//   crops: number[][];
// }

// typedef BinaryInputs = Pick<NamedTensorInfoMap, 'a'|'b'>;

class BinaryInputs {
  final TensorInfo a;
  final TensorInfo b;

  const BinaryInputs(this.a, this.b);
}

// export const Bincount = 'Bincount';
// export type BincountInputs = Pick<NamedTensorInfoMap, 'x'|'weights'>;
// export interface BincountAttrs {
//   size: number;
// }

// export const BroadcastTo = 'BroadcastTo';
// export type BroadcastToInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface BroadCastToAttrs {
//   shape: number[];
//   inputShape: number[];  // for gradient
// }

// export const BroadcastArgs = 'BroadcastArgs';
// export type BroadcastArgsInputs = Pick<NamedTensorInfoMap, 's0'|'s1'>;

const Cast = 'Cast';
typedef CastInputs = UnaryInputs;

class CastAttrs {
  final DataType dtype;

  CastAttrs(this.dtype);
}

const Ceil = 'Ceil';
typedef CeilInputs = UnaryInputs;

// export const ClipByValue = 'ClipByValue';
// export type ClipByValueInputs = UnaryInputs;
// export interface ClipByValueAttrs {
//   clipValueMin: number;
//   clipValueMax: number;
// }

const Complex = 'Complex';
// type ComplexInputs = Pick<NamedTensorInfoMap, 'real'|'imag'>;

const ComplexAbs = 'ComplexAbs';
typedef ComplexAbsInputs = UnaryInputs;

const Concat = 'Concat';
typedef ConcatInputs = List<TensorInfo>;
// interface ConcatAttrs {
//   axis: number;
// }

// export const Conv2D = 'Conv2D';
// export type Conv2DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
// export interface Conv2DAttrs {
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dataFormat: 'NHWC'|'NCHW';
//   dilations: [number, number]|number;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
// }

// export const Conv2DBackpropFilter = 'Conv2DBackpropFilter';
// export type Conv2DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x'|'dy'>;
// export interface Conv2DBackpropFilterAttrs {
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dataFormat: 'NHWC'|'NCHW';
//   dimRoundingMode?: 'floor'|'round'|'ceil';
//   filterShape: [number, number, number, number];
// }

// export const Conv2DBackpropInput = 'Conv2DBackpropInput';
// export type Conv2DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'|'filter'>;
// export interface Conv2DBackpropInputAttrs {
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dataFormat: 'NHWC'|'NCHW';
//   dimRoundingMode?: 'floor'|'round'|'ceil';
//   inputShape: [number, number, number, number];
// }

// export const Conv3D = 'Conv3D';
// export type Conv3DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
// export interface Conv3DAttrs {
//   strides: [number, number, number]|number;
//   pad: 'valid'|'same';
//   dataFormat: 'NDHWC'|'NCDHW';
//   dilations: [number, number, number]|number;
// }

// export const Conv3DBackpropFilterV2 = 'Conv3DBackpropFilterV2';
// export type Conv3DBackpropFilterV2Inputs = Pick<NamedTensorInfoMap, 'x'|'dy'>;

// export interface Conv3DBackpropFilterV2Attrs {
//   strides: [number, number, number]|number;
//   pad: 'valid'|'same';
//   filterShape: [number, number, number, number, number];
// }

// export const Conv3DBackpropInputV2 = 'Conv3DBackpropInputV2';
// export type Conv3DBackpropInputV2Inputs =
//     Pick<NamedTensorInfoMap, 'dy'|'filter'>;
// export interface Conv3DBackpropInputV2Attrs {
//   strides: [number, number, number]|number;
//   pad: 'valid'|'same';
//   inputShape: [number, number, number, number, number];
// }

const Cos = 'Cos';
typedef CosInputs = UnaryInputs;

const Cosh = 'Cosh';
typedef CoshInputs = UnaryInputs;

// export const Cumsum = 'Cumsum';
// export type CumsumInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface CumsumAttrs {
//   axis: number;
//   exclusive: boolean;
//   reverse: boolean;
// }

// export const CropAndResize = 'CropAndResize';
// export type CropAndResizeInputs =
//     Pick<NamedTensorInfoMap, 'image'|'boxes'|'boxInd'>;
// export interface CropAndResizeAttrs {
//   cropSize: [number, number];
//   method: 'bilinear'|'nearest';
//   extrapolationValue: number;
// }

// export const DenseBincount = 'DenseBincount';
// export type DenseBincountInputs = Pick<NamedTensorInfoMap, 'x'|'weights'>;
// export interface DenseBincountAttrs {
//   size: number;
//   binaryOutput?: boolean;
// }

// export const DepthToSpace = 'DepthToSpace';
// export type DepthToSpaceInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface DepthToSpaceAttrs {
//   blockSize: number;
//   dataFormat: 'NHWC'|'NCHW';
// }

const DepthwiseConv2dNative = 'DepthwiseConv2dNative';
typedef DepthwiseConv2dNativeInputs
    = NamedTensorInfoMap; // Pick<NamedTensorInfoMap, 'x'|'filter'>;

class DepthwiseConv2dNativeAttrs extends UnmodifiableMapBase<String, Object?>
    implements NamedAttrMap {
  final List<int> strides; //  [number, number]|number
  final String pad; //  'valid'|'same'|number|ExplicitPadding
  final String dataFormat; //  'NHWC'|'NCHW'
  final List<int> dilations; //  [number, number]|number
  final String? dimRoundingMode; // 'floor'|'round'|'ceil'

  DepthwiseConv2dNativeAttrs({
    required this.strides,
    required this.pad,
    required this.dataFormat,
    required this.dilations,
    required this.dimRoundingMode,
  });

  @override
  operator [](Object? key) {
    if (key is! String) return null;
    final index = keys.indexOf(key);
    if (index == -1) return null;
    return [strides, pad, dataFormat, dilations, dimRoundingMode][index];
  }

  @override
  List<String> get keys =>
      const ['strides', 'pad', 'dataFormat', 'dilations', 'dimRoundingMode'];
}

// export const DepthwiseConv2dNativeBackpropFilter =
//     'DepthwiseConv2dNativeBackpropFilter';
// export type DepthwiseConv2dNativeBackpropFilterInputs =
//     Pick<NamedTensorInfoMap, 'x'|'dy'>;
// export interface DepthwiseConv2dNativeBackpropFilterAttrs {
//   strides: [number, number]|number;
//   dilations: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
//   filterShape: [number, number, number, number];
// }

// export const DepthwiseConv2dNativeBackpropInput =
//     'DepthwiseConv2dNativeBackpropInput';
// export type DepthwiseConv2dNativeBackpropInputInputs =
//     Pick<NamedTensorInfoMap, 'dy'|'filter'>;
// export interface DepthwiseConv2dNativeBackpropInputAttrs {
//   strides: [number, number]|number;
//   dilations: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
//   inputShape: [number, number, number, number];
// }

// export const Diag = 'Diag';
// export type DiagInputs = Pick<NamedTensorInfoMap, 'x'>;

// export const Dilation2D = 'Dilation2D';
// export type Dilation2DInputs = Pick<NamedTensorInfoMap, 'x'|'filter'>;
// export interface Dilation2DAttrs {
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number;
//   dilations: [number, number]|number;
// }

// export const Dilation2DBackpropInput = 'Dilation2DBackpropInput';
// export type Dilation2DBackpropInputInputs =
//     Pick<NamedTensorInfoMap, 'x'|'filter'|'dy'>;

// export const Dilation2DBackpropFilter = 'Dilation2DBackpropFilter';
// export type Dilation2DBackpropFilterInputs =
//     Pick<NamedTensorInfoMap, 'x'|'filter'|'dy'>;

const RealDiv = 'RealDiv';
typedef RealDivInputs = BinaryInputs;

// export const Einsum = 'Einsum';
// export type EinsumInputs = TensorInfo[];
// export interface EinsumAttrs {
//   equation: string;
// }

const Elu = 'Elu';
typedef EluInputs = UnaryInputs;

// export const EluGrad = 'EluGrad';
// export type EluGradInputs = Pick<NamedTensorInfoMap, 'dy'|'y'>;

// export const Erf = 'Erf';
// export type ErfInputs = UnaryInputs;

const Equal = 'Equal';
typedef EqualInputs = BinaryInputs;

const Exp = 'Exp';
typedef ExpInputs = UnaryInputs;

const ExpandDims = 'ExpandDims';
// typedef ExpandDimsInputs = Pick<NamedTensorInfoMap, 'input'>;
// export interface ExpandDimsAttrs {
//   dim: number;
// }

// export const Expm1 = 'Expm1';
// export type Expm1Inputs = UnaryInputs;

// export const FFT = 'FFT';
// export type FFTInputs = Pick<NamedTensorInfoMap, 'input'>;

// export const Fill = 'Fill';
// export interface FillAttrs {
//   shape: number[];
//   value: number|string;
//   dtype: DataType;
// }

// export const FlipLeftRight = 'FlipLeftRight';
// export type FlipLeftRightInputs = Pick<NamedTensorInfoMap, 'image'>;

const Floor = 'Floor';
typedef FloorInputs = UnaryInputs;

const FloorDiv = 'FloorDiv';
typedef FloorDivInputs = BinaryInputs;

// export const FusedBatchNorm = 'FusedBatchNorm';
// export type FusedBatchNormInputs =
//     Pick<NamedTensorInfoMap, 'x'|'scale'|'offset'|'mean'|'variance'>;
// export interface FusedBatchNormAttrs {
//   varianceEpsilon: number;
// }

// export const GatherV2 = 'GatherV2';
// export type GatherV2Inputs = Pick<NamedTensorInfoMap, 'x'|'indices'>;
// export interface GatherV2Attrs {
//   axis: number;
//   batchDims: number;
// }

// export const GatherNd = 'GatherNd';
// export type GatherNdInputs = Pick<NamedTensorInfoMap, 'params'|'indices'>;

const Greater = 'Greater';
typedef GreaterInputs = BinaryInputs;

const GreaterEqual = 'GreaterEqual';
typedef GreaterEqualInputs = BinaryInputs;

const Identity = 'Identity';
typedef IdentityInputs = UnaryInputs; // Pick<NamedTensorInfoMap, 'x'>;

class UnaryInputs {
  final TensorInfo x;

  const UnaryInputs(this.x);
}

// export const IFFT = 'IFFT';
// export type IFFTInputs = Pick<NamedTensorInfoMap, 'input'>;

// export const Imag = 'Imag';
// export type ImagInputs = Pick<NamedTensorInfoMap, 'input'>;

// export const IsFinite = 'IsFinite';
// export type IsFiniteInputs = UnaryInputs;

// export const IsInf = 'IsInf';
// export type IsInfInputs = UnaryInputs;

// export const IsNan = 'IsNan';
// export type IsNanInputs = UnaryInputs;

// export const LeakyRelu = 'LeakyRelu';
// export type LeakyReluInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface LeakyReluAttrs {
//   alpha: number;
// }

const Less = 'Less';
typedef LessInputs = BinaryInputs;

const LessEqual = 'LessEqual';
typedef LessEqualInputs = BinaryInputs;

// export const LinSpace = 'LinSpace';
// export interface LinSpaceAttrs {
//   start: number;
//   stop: number;
//   num: number;
// }

const Log = 'Log';
typedef LogInputs = UnaryInputs;

// export const Log1p = 'Log1p';
// export type Log1pInputs = UnaryInputs;

const LogicalAnd = 'LogicalAnd';
typedef LogicalAndInputs = BinaryInputs;

const LogicalNot = 'LogicalNot';
typedef LogicalNotInputs = UnaryInputs;

const LogicalOr = 'LogicalOr';
typedef LogicalOrInputs = BinaryInputs;

// export const LogSoftmax = 'LogSoftmax';
// export type LogSoftmaxInputs = Pick<NamedTensorInfoMap, 'logits'>;
// export interface LogSoftmaxAttrs {
//   axis: number;
// }

// export const LRN = 'LRN';
// export type LRNInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface LRNAttrs {
//   depthRadius: number;
//   bias: number;
//   alpha: number;
//   beta: number;
// }

// export const LRNGrad = 'LRNGrad';
// export type LRNGradInputs = Pick<NamedTensorInfoMap, 'x'|'y'|'dy'>;
// export interface LRNGradAttrs {
//   depthRadius: number;
//   bias: number;
//   alpha: number;
//   beta: number;
// }

// export const Max = 'Max';
// export type MaxInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface MaxAttrs {
//   reductionIndices: number|number[];
//   keepDims: boolean;
// }

const Maximum = 'Maximum';
typedef MaximumInputs = BinaryInputs;

// export const MaxPool = 'MaxPool';
// export type MaxPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface MaxPoolAttrs {
//   filterSize: [number, number]|number;
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
// }

// export const MaxPoolGrad = 'MaxPoolGrad';
// export type MaxPoolGradInputs = Pick<NamedTensorInfoMap, 'dy'|'input'|'output'>;
// export interface MaxPoolGradAttrs {
//   filterSize: [number, number]|number;
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
// }

// export const MaxPool3D = 'MaxPool3D';
// export type MaxPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface MaxPool3DAttrs {
//   filterSize: [number, number, number]|number;
//   strides: [number, number, number]|number;
//   pad: 'valid'|'same'|number;
//   dataFormat: 'NDHWC'|'NCDHW';
//   dimRoundingMode?: 'floor'|'round'|'ceil';
// }

// export const MaxPool3DGrad = 'MaxPool3DGrad';
// export type MaxPool3DGradInputs =
//     Pick<NamedTensorInfoMap, 'dy'|'input'|'output'>;
// export interface MaxPool3DGradAttrs {
//   filterSize: [number, number, number]|number;
//   strides: [number, number, number]|number;
//   pad: 'valid'|'same'|number;
//   dimRoundingMode?: 'floor'|'round'|'ceil';
// }

// export const MaxPoolWithArgmax = 'MaxPoolWithArgmax';
// export type MaxPoolWithArgmaxInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface MaxPoolWithArgmaxAttrs {
//   filterSize: [number, number]|number;
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number;
//   includeBatchInIndex: boolean;
// }

// export const Mean = 'Mean';
// export type MeanInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface MeanAttrs {
//   axis: number|number[];
//   keepDims: boolean;
// }

// export const Min = 'Min';
// export type MinInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface MinAttrs {
//   axis: number|number[];
//   keepDims: boolean;
// }

const Minimum = 'Minimum';
typedef MinimumInputs = BinaryInputs;

// export const MirrorPad = 'MirrorPad';
// export type MirrorPadInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface MirrorPadAttrs {
//   paddings: Array<[number, number]>;
//   mode: 'reflect'|'symmetric';
// }

// TODO:
// const Mod = 'Mod';
// typedef ModInputs = BinaryInputs;

// export const Multinomial = 'Multinomial';
// export type MultinomialInputs = Pick<NamedTensorInfoMap, 'logits'>;
// export interface MultinomialAttrs {
//   numSamples: number;
//   seed: number;
//   normalized: boolean;
// }

const Multiply = 'Multiply';
typedef MultiplyInputs = BinaryInputs;

const Neg = 'Neg';
typedef NegInputs = UnaryInputs;

const NotEqual = 'NotEqual';
typedef NotEqualInputs = BinaryInputs;

// export const NonMaxSuppressionV3 = 'NonMaxSuppressionV3';
// export type NonMaxSuppressionV3Inputs =
//     Pick<NamedTensorInfoMap, 'boxes'|'scores'>;
// export interface NonMaxSuppressionV3Attrs {
//   maxOutputSize: number;
//   iouThreshold: number;
//   scoreThreshold: number;
// }

// export const NonMaxSuppressionV4 = 'NonMaxSuppressionV4';
// export type NonMaxSuppressionV4Inputs =
//     Pick<NamedTensorInfoMap, 'boxes'|'scores'>;
// export interface NonMaxSuppressionV4Attrs {
//   maxOutputSize: number;
//   iouThreshold: number;
//   scoreThreshold: number;
//   padToMaxOutputSize: boolean;
// }

// export const NonMaxSuppressionV5 = 'NonMaxSuppressionV5';
// export type NonMaxSuppressionV5Inputs =
//     Pick<NamedTensorInfoMap, 'boxes'|'scores'>;
// export interface NonMaxSuppressionV5Attrs {
//   maxOutputSize: number;
//   iouThreshold: number;
//   scoreThreshold: number;
//   softNmsSigma: number;
// }

// export const OnesLike = 'OnesLike';
// export type OnesLikeInputs = UnaryInputs;

// export const OneHot = 'OneHot';
// export type OneHotInputs = Pick<NamedTensorInfoMap, 'indices'>;
// export interface OneHotAttrs {
//   depth: number;
//   onValue: number;
//   offValue: number;
// }

const Pack = 'Pack';
typedef PackInputs = UnaryInputs; // TODO: TensorInfo[];
// interface PackAttrs {
//   axis: number;
// }

// export const PadV2 = 'PadV2';
// export type PadV2Inputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface PadV2Attrs {
//   paddings: Array<[number, number]>;
//   constantValue: number;
// }

// export const Pool = 'Pool';
// export type PoolInputs = Pick<NamedTensorInfoMap, 'input'>;

const Pow = 'Pow';
typedef PowInputs = BinaryInputs;

// export const Prelu = 'Prelu';
// export type PreluInputs = Pick<NamedTensorInfoMap, 'x'|'alpha'>;

// export const Prod = 'Prod';
// export type ProdInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface ProdAttrs {
//   axis: number|number[];
//   keepDims: boolean;
// }

// export const Range = 'Range';
// export interface RangeAttrs {
//   start: number;
//   stop: number;
//   step: number;
//   dtype: 'float32'|'int32';
// }

// export const Real = 'Real';
// export type RealInputs = Pick<NamedTensorInfoMap, 'input'>;

const Reciprocal = 'Reciprocal';
typedef ReciprocalInputs = UnaryInputs;

const Relu = 'Relu';
typedef ReluInputs = UnaryInputs;

const Reshape = 'Reshape';
typedef ReshapeInputs = UnaryInputs;
// export interface ReshapeAttrs {
//   shape: number[];
// }

// export const ResizeNearestNeighbor = 'ResizeNearestNeighbor';
// export type ResizeNearestNeighborInputs = Pick<NamedTensorInfoMap, 'images'>;
// export interface ResizeNearestNeighborAttrs {
//   alignCorners: boolean;
//   halfPixelCenters: boolean;
//   size: [number, number];
// }

// export const ResizeNearestNeighborGrad = 'ResizeNearestNeighborGrad';
// export type ResizeNearestNeighborGradInputs =
//     Pick<NamedTensorInfoMap, 'images'|'dy'>;
// export type ResizeNearestNeighborGradAttrs = ResizeNearestNeighborAttrs;

// export const ResizeBilinear = 'ResizeBilinear';
// export type ResizeBilinearInputs = Pick<NamedTensorInfoMap, 'images'>;
// export interface ResizeBilinearAttrs {
//   alignCorners: boolean;
//   halfPixelCenters: boolean;
//   size: [number, number];
// }

// export const ResizeBilinearGrad = 'ResizeBilinearGrad';
// export type ResizeBilinearGradInputs = Pick<NamedTensorInfoMap, 'images'|'dy'>;
// export type ResizeBilinearGradAttrs = ResizeBilinearAttrs;

const Relu6 = 'Relu6';
typedef Relu6Inputs = UnaryInputs;

// export const Reverse = 'Reverse';
// export type ReverseInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface ReverseAttrs {
//   dims: number|number[];
// }

const Round = 'Round';
typedef RoundInputs = UnaryInputs;

const Rsqrt = 'Rsqrt';
typedef RsqrtInputs = UnaryInputs;

// export const ScatterNd = 'ScatterNd';
// export type ScatterNdInputs = Pick<NamedTensorInfoMap, 'indices'|'updates'>;
// export interface ScatterNdAttrs {
//   shape: number[];
// }

// export const Select = 'Select';
// export type SelectInputs = Pick<NamedTensorInfoMap, 'condition'|'t'|'e'>;

// export const Selu = 'Selu';
// export type SeluInputs = Pick<NamedTensorInfoMap, 'x'>;

const Slice = 'Slice';
typedef SliceInputs = UnaryInputs; // Pick<NamedTensorInfoMap, 'x'>;
// export interface SliceAttrs {
//   begin: number|number[];
//   size: number|number[];
// }
const Sin = 'Sin';
typedef SinInputs = UnaryInputs;

// export const Sinh = 'Sinh';
// export type SinhInputs = UnaryInputs;

const Sign = 'Sign';
typedef SignInputs = UnaryInputs;

// export const Sigmoid = 'Sigmoid';
// export type SigmoidInputs = UnaryInputs;

// export const Softplus = 'Softplus';
// export type SoftplusInputs = UnaryInputs;

const Sqrt = 'Sqrt';
typedef SqrtInputs = UnaryInputs;

// export const Sum = 'Sum';
// export type SumInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface SumAttrs {
//   axis: number|number[];
//   keepDims: boolean;
// }

// export const SpaceToBatchND = 'SpaceToBatchND';
// export type SpaceToBatchNDInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface SpaceToBatchNDAttrs {
//   blockShape: number[];
//   paddings: number[][];
// }

// export const SplitV = 'SplitV';
// export type SplitVInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface SplitVAttrs {
//   numOrSizeSplits: number[]|number;
//   axis: number;
// }

// export const Softmax = 'Softmax';
// export type SoftmaxInputs = Pick<NamedTensorInfoMap, 'logits'>;
// export interface SoftmaxAttrs {
//   dim: number;
// }

// export const SparseFillEmptyRows = 'SparseFillEmptyRows';
// export type SparseFillEmptyRowsInputs =
//     Pick<NamedTensorInfoMap, 'indices'|'values'|'denseShape'|'defaultValue'>;

// export const SparseReshape = 'SparseReshape';
// export type SparseReshapeInputs =
//     Pick<NamedTensorInfoMap, 'inputIndices'|'inputShape'|'newShape'>;

// export const SparseSegmentMean = 'SparseSegmentMean';
// export type SparseSegmentMeanInputs =
//     Pick<NamedTensorInfoMap, 'data'|'indices'|'segmentIds'>;

// export const SparseSegmentSum = 'SparseSegmentSum';
// export type SparseSegmentSumInputs =
//     Pick<NamedTensorInfoMap, 'data'|'indices'|'segmentIds'>;

// export const SparseToDense = 'SparseToDense';
// export type SparseToDenseInputs =
//     Pick<NamedTensorInfoMap, 'sparseIndices'|'sparseValues'|'defaultValue'>;
// export interface SparseToDenseAttrs {
//   outputShape: number[];
// }

const SquaredDifference = 'SquaredDifference';
typedef SquaredDifferenceInputs = BinaryInputs;

const Square = 'Square';
typedef SquareInputs = UnaryInputs;

// export const StridedSlice = 'StridedSlice';
// export type StridedSliceInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface StridedSliceAttrs {
//   begin: number[];
//   end: number[];
//   strides: number[];
//   beginMask: number;
//   endMask: number;
//   ellipsisMask: number;
//   newAxisMask: number;
//   shrinkAxisMask: number;
// }

// export const StringNGrams = 'StringNGrams';
// export type StringNGramsInputs = Pick<NamedTensorInfoMap, 'data'|'dataSplits'>;
// export interface StringNGramsAttrs {
//   separator: string;
//   nGramWidths: number[];
//   leftPad: string;
//   rightPad: string;
//   padWidth: number;
//   preserveShortSequences: boolean;
// }

// export const StringSplit = 'StringSplit';
// export type StringSplitInputs = Pick<NamedTensorInfoMap, 'input'|'delimiter'>;
// export interface StringSplitAttrs {
//   skipEmpty: boolean;
// }

// export const StringToHashBucketFast = 'StringToHashBucketFast';
// export type StringToHashBucketFastInputs = Pick<NamedTensorInfoMap, 'input'>;
// export interface StringToHashBucketFastAttrs {
//   numBuckets: number;
// }

const Sub = 'Sub';
typedef SubInputs = BinaryInputs;

const Tan = 'Tan';
typedef TanInputs = UnaryInputs;

const Tanh = 'Tanh';
typedef TanhInputs = UnaryInputs;

// export const Tile = 'Tile';
// export type TileInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface TileAttrs {
//   reps: number[];
// }

// export const TopK = 'TopK';
// export type TopKInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface TopKAttrs {
//   k: number;
//   sorted: boolean;
// }

// export const Transform = 'Transform';
// export type TransformInputs = Pick<NamedTensorInfoMap, 'image'|'transforms'>;
// export interface TransformAttrs {
//   interpolation: 'nearest'|'bilinear';
//   fillMode: 'constant'|'reflect'|'wrap'|'nearest';
//   fillValue: number;
//   outputShape?: [number, number];
// }

const Transpose = 'Transpose';
typedef TransposeInputs = UnaryInputs; // Pick<NamedTensorInfoMap, 'x'>;
// export interface TransposeAttrs {
//   perm: number[];
// }

// export const Unique = 'Unique';
// export type UniqueInputs = Pick<NamedTensorInfoMap, 'x'>;
// export interface UniqueAttrs {
//   axis: number;
// }

// export type UnaryInputs = Pick<NamedTensorInfoMap, 'x'>;

const Unpack = 'Unpack';
typedef UnpackInputs
    = Map<String, Tensor>; // Pick<NamedTensorInfoMap, 'value'>;
// interface UnpackAttrs {
//   axis: number;
// }

// export const UnsortedSegmentSum = 'UnsortedSegmentSum';
// export type UnsortedSegmentSumInputs =
//     Pick<NamedTensorInfoMap, 'x'|'segmentIds'>;
// export interface UnsortedSegmentSumAttrs {
//   numSegments: number;
// }

// export const ZerosLike = 'ZerosLike';
// export type ZerosLikeInputs = UnaryInputs;

// /**
//  * TensorFlow.js-only kernels
//  */
// export const Step = 'Step';
// export type StepInputs = UnaryInputs;
// export interface StepAttrs {
//   alpha: number;
// }

// export const FromPixels = 'FromPixels';
// export interface FromPixelsInputs {
//   pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
//       HTMLVideoElement|ImageBitmap;
// }
// export interface FromPixelsAttrs {
//   numChannels: number;
// }

// export const RotateWithOffset = 'RotateWithOffset';
// export type RotateWithOffsetInputs = Pick<NamedTensorInfoMap, 'image'>;
// export interface RotateWithOffsetAttrs {
//   radians: number;
//   fillValue: number|[number, number, number];
//   center: number|[number, number];
// }

// export const _FusedMatMul = '_FusedMatMul';
// // tslint:disable-next-line: class-name
// export interface _FusedMatMulInputs extends NamedTensorInfoMap {
//   a: TensorInfo;
//   b: TensorInfo;
//   bias?: TensorInfo;
//   preluActivationWeights?: TensorInfo;
// }
// // tslint:disable-next-line: class-name
// export interface _FusedMatMulAttrs {
//   transposeA: boolean;
//   transposeB: boolean;
//   activation: Activation;
//   leakyreluAlpha?: number;
// }

// export const FusedConv2D = 'FusedConv2D';
// export interface FusedConv2DInputs extends NamedTensorInfoMap {
//   x: TensorInfo;
//   filter: TensorInfo;
//   bias?: TensorInfo;
//   preluActivationWeights?: TensorInfo;
// }
// export interface FusedConv2DAttrs {
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dataFormat: 'NHWC'|'NCHW';
//   dilations: [number, number]|number;
//   dimRoundingMode: 'floor'|'round'|'ceil';
//   activation: Activation;
//   leakyreluAlpha?: number;
// }

// export const FusedDepthwiseConv2D = 'FusedDepthwiseConv2D';
// export interface FusedDepthwiseConv2DInputs extends NamedTensorInfoMap {
//   x: TensorInfo;
//   filter: TensorInfo;
//   bias?: TensorInfo;
//   preluActivationWeights?: TensorInfo;
// }
// export interface FusedDepthwiseConv2DAttrs {
//   strides: [number, number]|number;
//   pad: 'valid'|'same'|number|ExplicitPadding;
//   dataFormat: 'NHWC'|'NCHW';
//   dilations: [number, number]|number;
//   dimRoundingMode: 'floor'|'round'|'ceil';
//   activation: Activation;
//   leakyreluAlpha?: number;
// }
