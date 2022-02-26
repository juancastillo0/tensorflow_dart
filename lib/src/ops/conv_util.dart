// /**
//  * @license
//  * Copyright 2020 Google LLC. All Rights Reserved.
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  * =============================================================================
//  */

// import * as util from '../util';

import 'dart:math' as math;

import '../util_base.dart' as util;

enum PadType {
  SAME,
  VALID,
  NUMBER,
  EXPLICIT,
}

// // For NHWC should be in the following form:
// //  [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]
// // For NCHW should be in the following form:
// //  [[0, 0], [0, 0], [pad_top,pad_bottom], [pad_left, pad_right]]
// // Reference: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
// export type ExplicitPadding =
//     [[number, number], [number, number], [number, number], [number, number]];

class PadInfo {
  final int top; //: number,
  final int left; //: number,
  final int right; //: number,
  final int bottom; //: number,
  final PadType type;

  const PadInfo({
    required this.top,
    required this.left,
    required this.right,
    required this.bottom,
    required this.type,
  }); //: PadType

}

// export type PadInfo3D = {
//   top: number,
//   left: number,
//   right: number,
//   bottom: number,
//   front: number,
//   back: number,
//   type: PadType
// };

/**
 * Information about the forward pass of a convolution/pooling operation.
 * It includes input and output shape, strides, filter size and padding
 * information.
 */
class Conv2DInfo {
  final int batchSize; //: number,
  final int inHeight; //: number,
  final int inWidth; //: number,
  final int inChannels; //: number,
  final int outHeight; //: number,
  final int outWidth; //: number,
  final int outChannels; //: number,
  final String dataFormat; //: 'channelsFirst'|'channelsLast',
  final int strideHeight; //: number,
  final int strideWidth; //: number,
  final int dilationHeight; //: number,
  final int dilationWidth; //: number,
  final int filterHeight; //: number,
  final int filterWidth; //: number,
  final int effectiveFilterHeight; //: number,
  final int effectiveFilterWidth; //: number,
  final PadInfo padInfo; //: PadInfo,
  final List<int> inShape; //: [number, number, number, number],
  final List<int> outShape; //: [number, number, number, number],
  final List<int> filterShape; //: [number, number, number, number]

  Conv2DInfo({
    required this.batchSize,
    required this.inHeight,
    required this.inWidth,
    required this.inChannels,
    required this.outHeight,
    required this.outWidth,
    required this.outChannels,
    required this.dataFormat,
    required this.strideHeight,
    required this.strideWidth,
    required this.dilationHeight,
    required this.dilationWidth,
    required this.filterHeight,
    required this.filterWidth,
    required this.effectiveFilterHeight,
    required this.effectiveFilterWidth,
    required this.padInfo,
    required this.inShape,
    required this.outShape,
    required this.filterShape,
  });
}

// /**
//  *
//  * @param inputShape Input tensor shape is of the following dimensions:
//  *     `[batch, height, width, inChannels]`.
//  * @param filterShape The filter shape is of the following dimensions:
//  *     `[filterHeight, filterWidth, depth]`.
//  * @param strides The strides of the sliding window for each dimension of the
//  *     input tensor: `[strideHeight, strideWidth]`.
//  *     If `strides` is a single number,
//  *     then `strideHeight == strideWidth`.
//  * @param pad The type of padding algorithm.
//  *    - `same` and stride 1: output will be of same size as input,
//  *       regardless of filter size.
//  *    - `valid`: output will be smaller than input if filter is larger
//  *       than 1*1x1.
//  *    - For more info, see this guide:
//  *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
//  *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
//  * @param dataFormat The data format of the input and output data.
//  *     Defaults to 'NHWC'.
//  * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`.
//  *     Defaults to `[1, 1]`. If `dilations` is a single number, then
//  *     `dilationHeight == dilationWidth`.
//  */
// export function computeDilation2DInfo(
//     inputShape: [number, number, number, number],
//     filterShape: [number, number, number], strides: number|[number, number],
//     pad: 'same'|'valid'|number, dataFormat: 'NHWC' = 'NHWC',
//     dilations: number|[number, number]) {
//   // `computerConv2DInfo` require filterShape to be in the dimension of:
//   // `[filterHeight, filterWidth, depth, outDepth]`, dilation2d doesn't have
//   // outDepth, it should have the same depth as the input.
//   // Input shape: [batch, height, width, inChannels]
//   const inputChannels = inputShape[3];
//   const $filterShape =
//       [...filterShape, inputChannels] as [number, number, number, number];
//   const $dataFormat = convertConv2DDataFormat(dataFormat);

//   return computeConv2DInfo(
//       inputShape, $filterShape, strides, dilations, pad,
//       null /* roundingMode */, null /* depthWise */, $dataFormat);
// }

// export function computePool2DInfo(
//     inShape: [number, number, number, number],
//     filterSize: [number, number]|number, strides: number|[number, number],
//     dilations: number|[number, number],
//     pad: 'same'|'valid'|number|ExplicitPadding,
//     roundingMode?: 'floor'|'round'|'ceil',
//     dataFormat: 'channelsFirst'|'channelsLast' = 'channelsLast'): Conv2DInfo {
//   const [filterHeight, filterWidth] = parseTupleParam(filterSize);

//   let filterShape: [number, number, number, number];
//   if (dataFormat === 'channelsLast') {
//     filterShape = [filterHeight, filterWidth, inShape[3], inShape[3]];
//   } else if (dataFormat === 'channelsFirst') {
//     filterShape = [filterHeight, filterWidth, inShape[1], inShape[1]];
//   } else {
//     throw Exception(`Unknown dataFormat ${dataFormat}`);
//   }

//   return computeConv2DInfo(
//       inShape, filterShape, strides, dilations, pad, roundingMode, false,
//       dataFormat);
// }

// /**
//  * Computes the information for a forward pass of a pooling3D operation.
//  */
// export function computePool3DInfo(
//     inShape: [number, number, number, number, number],
//     filterSize: number|[number, number, number],
//     strides: number|[number, number, number],
//     dilations: number|[number, number, number], pad: 'same'|'valid'|number,
//     roundingMode?: 'floor'|'round'|'ceil',
//     dataFormat: 'NDHWC'|'NCDHW' = 'NDHWC'): Conv3DInfo {
//   const [filterDepth, filterHeight, filterWidth] = parse3TupleParam(filterSize);

//   let filterShape: [number, number, number, number, number];
//   let $dataFormat: 'channelsFirst'|'channelsLast';
//   if (dataFormat === 'NDHWC') {
//     $dataFormat = 'channelsLast';
//     filterShape =
//         [filterDepth, filterHeight, filterWidth, inShape[4], inShape[4]];
//   } else if (dataFormat === 'NCDHW') {
//     $dataFormat = 'channelsFirst';
//     filterShape =
//         [filterDepth, filterHeight, filterWidth, inShape[1], inShape[1]];
//   } else {
//     throw Exception(`Unknown dataFormat ${dataFormat}`);
//   }

//   return computeConv3DInfo(
//       inShape, filterShape, strides, dilations, pad, false, $dataFormat,
//       roundingMode);
// }

/**
 * Computes the information for a forward pass of a convolution/pooling
 * operation.
 */
Conv2DInfo computeConv2DInfo(
  List<int> inShape, // : [number, number, number, number],
  List<int> filterShape, // : [number, number, number, number],
  List<int> strides, // : number|[number, number],
  List<int> dilations, // : number|[number, number],
  {
  required Object pad, // : 'same'|'valid'|number|ExplicitPadding,
  String? roundingMode, // ?: 'floor'|'round'|'ceil',
  bool depthwise = false,
  // : 'channelsFirst'|'channelsLast'
  String dataFormat = 'channelsLast',
}) {
  int batchSize;
  int inHeight;
  int inWidth;
  int inChannels;
  if (dataFormat == 'channelsLast') {
    batchSize = inShape[0];
    inHeight = inShape[1];
    inWidth = inShape[2];
    inChannels = inShape[3];
  } else if (dataFormat == 'channelsFirst') {
    batchSize = inShape[0];
    inHeight = inShape[2];
    inWidth = inShape[3];
    inChannels = inShape[1];
  } else {
    throw Exception('Unknown dataFormat ${dataFormat}');
  }

  final filterHeight = filterShape[0];
  final filterWidth = filterShape[1];
  final filterChannels = filterShape[3];

  final _s = parseTupleParam(strides);
  final strideHeight = _s[0];
  final strideWidth = _s[1];
  final _d = parseTupleParam(dilations);
  final dilationHeight = _d[0];
  final dilationWidth = _d[1];

  final effectiveFilterHeight =
      _getEffectiveFilterSize(filterHeight, dilationHeight);
  final effectiveFilterWidth =
      _getEffectiveFilterSize(filterWidth, dilationWidth);
  final _po = _getPadAndOutInfo(
    pad,
    inHeight,
    inWidth,
    strideHeight,
    strideWidth,
    effectiveFilterHeight,
    effectiveFilterWidth,
    roundingMode,
    dataFormat,
  );
  final padInfo = _po.padInfo;
  final outHeight = _po.outHeight;
  final outWidth = _po.outWidth;

  final outChannels = depthwise ? filterChannels * inChannels : filterChannels;

  late List<int> outShape;
  if (dataFormat == 'channelsFirst') {
    outShape = [batchSize, outChannels, outHeight, outWidth];
  } else if (dataFormat == 'channelsLast') {
    outShape = [batchSize, outHeight, outWidth, outChannels];
  }

  return Conv2DInfo(
    batchSize: batchSize,
    dataFormat: dataFormat,
    inHeight: inHeight,
    inWidth: inWidth,
    inChannels: inChannels,
    outHeight: outHeight,
    outWidth: outWidth,
    outChannels: outChannels,
    padInfo: padInfo,
    strideHeight: strideHeight,
    strideWidth: strideWidth,
    filterHeight: filterHeight,
    filterWidth: filterWidth,
    effectiveFilterHeight: effectiveFilterHeight,
    effectiveFilterWidth: effectiveFilterWidth,
    dilationHeight: dilationHeight,
    dilationWidth: dilationWidth,
    inShape: inShape,
    outShape: outShape,
    filterShape: filterShape,
  );
}

// /**
//  * Information about the forward pass of a 3D convolution/pooling operation.
//  * It includes input and output shape, strides, filter size and padding
//  * information.
//  */
// export type Conv3DInfo = {
//   batchSize: number,
//   inDepth: number,
//   inHeight: number,
//   inWidth: number,
//   inChannels: number,
//   outDepth: number,
//   outHeight: number,
//   outWidth: number,
//   outChannels: number,
//   dataFormat: 'channelsFirst'|'channelsLast',
//   strideDepth: number,
//   strideHeight: number,
//   strideWidth: number,
//   dilationDepth: number,
//   dilationHeight: number,
//   dilationWidth: number,
//   filterDepth: number,
//   filterHeight: number,
//   filterWidth: number,
//   effectiveFilterDepth: number,
//   effectiveFilterHeight: number,
//   effectiveFilterWidth: number,
//   padInfo: PadInfo3D,
//   inShape: [number, number, number, number, number],
//   outShape: [number, number, number, number, number],
//   filterShape: [number, number, number, number, number]
// };

// /**
//  * Computes the information for a forward pass of a 3D convolution/pooling
//  * operation.
//  */
// export function computeConv3DInfo(
//     inShape: [number, number, number, number, number],
//     filterShape: [number, number, number, number, number],
//     strides: number|[number, number, number],
//     dilations: number|[number, number, number], pad: 'same'|'valid'|number,
//     depthwise = false,
//     dataFormat: 'channelsFirst'|'channelsLast' = 'channelsLast',
//     roundingMode?: 'floor'|'round'|'ceil'): Conv3DInfo {
//   let [batchSize, inDepth, inHeight, inWidth, inChannels] =
//       [-1, -1, -1, -1, -1];
//   if (dataFormat === 'channelsLast') {
//     [batchSize, inDepth, inHeight, inWidth, inChannels] = inShape;
//   } else if (dataFormat === 'channelsFirst') {
//     [batchSize, inChannels, inDepth, inHeight, inWidth] = inShape;
//   } else {
//     throw Exception(`Unknown dataFormat ${dataFormat}`);
//   }

//   const [filterDepth, filterHeight, filterWidth, , filterChannels] =
//       filterShape;
//   const [strideDepth, strideHeight, strideWidth] = parse3TupleParam(strides);
//   const [dilationDepth, dilationHeight, dilationWidth] =
//       parse3TupleParam(dilations);

//   const effectiveFilterDepth =
//       _getEffectiveFilterSize(filterDepth, dilationDepth);
//   const effectiveFilterHeight =
//       _getEffectiveFilterSize(filterHeight, dilationHeight);
//   const effectiveFilterWidth =
//       _getEffectiveFilterSize(filterWidth, dilationWidth);
//   const {padInfo, outDepth, outHeight, outWidth} = get3DPadAndOutInfo(
//       pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth,
//       effectiveFilterDepth, effectiveFilterHeight, effectiveFilterWidth,
//       roundingMode);

//   const outChannels = depthwise ? filterChannels * inChannels : filterChannels;

//   let outShape: [number, number, number, number, number];
//   if (dataFormat === 'channelsFirst') {
//     outShape = [batchSize, outChannels, outDepth, outHeight, outWidth];
//   } else if (dataFormat === 'channelsLast') {
//     outShape = [batchSize, outDepth, outHeight, outWidth, outChannels];
//   }

//   return {
//     batchSize,
//     dataFormat,
//     inDepth,
//     inHeight,
//     inWidth,
//     inChannels,
//     outDepth,
//     outHeight,
//     outWidth,
//     outChannels,
//     padInfo,
//     strideDepth,
//     strideHeight,
//     strideWidth,
//     filterDepth,
//     filterHeight,
//     filterWidth,
//     effectiveFilterDepth,
//     effectiveFilterHeight,
//     effectiveFilterWidth,
//     dilationDepth,
//     dilationHeight,
//     dilationWidth,
//     inShape,
//     outShape,
//     filterShape
//   };
// }
class Size {
  final int height;
  final int width;
  Size({
    required this.height,
    required this.width,
  });
}

Size _computeOutputShape2D(
  List<int> inShape,
  int fieldSize,
  int stride, {
  int? zeroPad,
  String? roundingMode,
}
    //  ?: 'floor'|'round'|'ceil'
    ) {
  zeroPad ??= computeDefaultPad(inShape, fieldSize, stride);

  final inputRows = inShape[0];
  final inputCols = inShape[1];

  final outputRows =
      round((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  final outputCols =
      round((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);

  return Size(height: outputRows, width: outputCols);
}

// function computeOutputShape4D(
//     inShape: [number, number, number, number], fieldSize: number,
//     outChannels: number, stride: number, zeroPad?: number,
//     roundingMode?: 'floor'|'round'|'ceil'): [number, number, number, number] {
//   if (zeroPad == null) {
//     zeroPad = computeDefaultPad(inShape, fieldSize, stride);
//   }
//   const inputDepth = inShape[0];
//   const inputRows = inShape[1];
//   const inputCols = inShape[2];

//   const outputDepths =
//       round((inputDepth - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
//   const outputRows =
//       round((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
//   const outputCols =
//       round((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);

//   return [outputDepths, outputRows, outputCols, outChannels];
// }

int computeDefaultPad(
  List<int> inputShape,
  // : [number, number]|[number, number, number, number],
  int fieldSize,
  int stride, [
  int dilation = 1,
]) {
  final effectiveFieldSize = _getEffectiveFilterSize(fieldSize, dilation);
  return ((inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2)
      .floor();
}

List<int> parseTupleParam(
    List<int> // : number|number[]
        param)
// : [number, number, number]
{
  if (param is int || param.length == 1) {
    final _p = param is int ? param as int : param[0];
    return [_p, _p, _p];
  }
  if (param.length == 2) {
    return [param[0], param[1], 1];
  }
  return param;
}

// function parse3TupleParam(param: number|[number, number, number]):
//     [number, number, number] {
//   return typeof param === 'number' ? [param, param, param] : param;
// }

/* See https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
 * Atrous convolution is equivalent to standard convolution with upsampled
 * filters with effective_filter_height =
 * filter_height + (filter_height - 1) * (dilation - 1)
 * and effective_filter_width =
 * filter_width + (filter_width - 1) * (dilation - 1),
 * produced by inserting dilation - 1 zeros along consecutive elements across
 * the filters' spatial dimensions.
 * When there is a dilation, this converts a filter dimension to the
 * effective filter dimension, so it can be used in a standard convolution.
 */
int _getEffectiveFilterSize(int filterSize, int dilation) {
  if (dilation <= 1) {
    return filterSize;
  }

  return filterSize + (filterSize - 1) * (dilation - 1);
}

class _PadAndOutInfo {
  final PadInfo padInfo;
  final int outHeight;
  final int outWidth;

  _PadAndOutInfo({
    required this.padInfo,
    required this.outHeight,
    required this.outWidth,
  });
}

_PadAndOutInfo _getPadAndOutInfo(
  Object pad, // : 'same'|'valid'|number|ExplicitPadding,
  int inHeight, // : number,
  int inWidth, // : number,
  int strideHeight, // : number,
  int strideWidth, // : number,
  int filterHeight, // : number,
  int filterWidth, // : number,
  String? roundingMode, // : 'floor'|'round'|'ceil',
  String dataFormat, // : 'channelsFirst'|'channelsLast',
) {
  PadInfo padInfo;
  int outHeight;
  int outWidth;

  if (pad is int) {
    final padType = (pad == 0) ? PadType.VALID : PadType.NUMBER;
    padInfo =
        PadInfo(top: pad, bottom: pad, left: pad, right: pad, type: padType);
    final outShape = _computeOutputShape2D(
        [inHeight, inWidth], filterHeight, strideHeight,
        zeroPad: pad, roundingMode: roundingMode);
    outHeight = outShape.height;
    outWidth = outShape.width;
  } else if (pad == 'same') {
    outHeight = (inHeight / strideHeight).ceil();
    outWidth = (inWidth / strideWidth).ceil();
    final padAlongHeight =
        math.max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
    final padAlongWidth =
        math.max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
    final top = (padAlongHeight / 2).floor();
    final bottom = padAlongHeight - top;
    final left = (padAlongWidth / 2).floor();
    final right = padAlongWidth - left;
    padInfo = PadInfo(
        top: top, bottom: bottom, left: left, right: right, type: PadType.SAME);
  } else if (pad == 'valid') {
    padInfo =
        PadInfo(top: 0, bottom: 0, left: 0, right: 0, type: PadType.VALID);
    outHeight = ((inHeight - filterHeight + 1) / strideHeight).ceil();
    outWidth = ((inWidth - filterWidth + 1) / strideWidth).ceil();
  } else if (pad is List<List<int>>) {
    final top = dataFormat == 'channelsLast' ? pad[1][0] : pad[2][0];
    final bottom = dataFormat == 'channelsLast' ? pad[1][1] : pad[2][1];
    final left = dataFormat == 'channelsLast' ? pad[2][0] : pad[3][0];
    final right = dataFormat == 'channelsLast' ? pad[2][1] : pad[3][1];
    final padType = (top == 0 && bottom == 0 && left == 0 && right == 0)
        ? PadType.VALID
        : PadType.EXPLICIT;
    padInfo = PadInfo(
        top: top, bottom: bottom, left: left, right: right, type: padType);
    outHeight = round(
        (inHeight - filterHeight + top + bottom) / strideHeight + 1,
        roundingMode);
    outWidth = round(
        (inWidth - filterWidth + left + right) / strideWidth + 1, roundingMode);
  } else {
    throw Exception('Unknown padding parameter: ${pad}');
  }
  return _PadAndOutInfo(
    padInfo: padInfo,
    outHeight: outHeight,
    outWidth: outWidth,
  );
}

// function get3DPadAndOutInfo(
//     pad: 'same'|'valid'|number, inDepth: number, inHeight: number,
//     inWidth: number, strideDepth: number, strideHeight: number,
//     strideWidth: number, filterDepth: number, filterHeight: number,
//     filterWidth: number, roundingMode?: 'floor'|'round'|'ceil'): {
//   padInfo: PadInfo3D,
//   outDepth: number,
//   outHeight: number,
//   outWidth: number
// } {
//   let padInfo: PadInfo3D;
//   let outDepth: number;
//   let outHeight: number;
//   let outWidth: number;

//   if (typeof pad === 'number') {
//     const padType = (pad === 0) ? 'VALID' : 'NUMBER';
//     padInfo = {
//       top: pad,
//       bottom: pad,
//       left: pad,
//       right: pad,
//       front: pad,
//       back: pad,
//       type: padType
//     };
//     const outShape = computeOutputShape4D(
//         [inDepth, inHeight, inWidth, 1], filterDepth, 1, strideDepth, pad,
//         roundingMode);
//     outDepth = outShape[0];
//     outHeight = outShape[1];
//     outWidth = outShape[2];
//   } else if (pad === 'same') {
//     outDepth = math.ceil(inDepth / strideDepth);
//     outHeight = math.ceil(inHeight / strideHeight);
//     outWidth = math.ceil(inWidth / strideWidth);
//     const padAlongDepth = (outDepth - 1) * strideDepth + filterDepth - inDepth;
//     const padAlongHeight =
//         (outHeight - 1) * strideHeight + filterHeight - inHeight;
//     const padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
//     const front = math.floor(padAlongDepth / 2);
//     const back = padAlongDepth - front;
//     const top = math.floor(padAlongHeight / 2);
//     const bottom = padAlongHeight - top;
//     const left = math.floor(padAlongWidth / 2);
//     const right = padAlongWidth - left;

//     padInfo = {top, bottom, left, right, front, back, type: 'SAME'};
//   } else if (pad === 'valid') {
//     padInfo = {
//       top: 0,
//       bottom: 0,
//       left: 0,
//       right: 0,
//       front: 0,
//       back: 0,
//       type: 'VALID'
//     };
//     outDepth = math.ceil((inDepth - filterDepth + 1) / strideDepth);
//     outHeight = math.ceil((inHeight - filterHeight + 1) / strideHeight);
//     outWidth = math.ceil((inWidth - filterWidth + 1) / strideWidth);
//   } else {
//     throw Error(`Unknown padding parameter: ${pad}`);
//   }
//   return {padInfo, outDepth, outHeight, outWidth};
// }

/**
 * Rounds a value depending on the rounding mode
 * @param value
 * @param roundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
int round(num value, [String? roundingMode //?: 'floor'|'round'|'ceil'
    ]) {
  if (roundingMode == null) {
    return value.truncate();
  }
  switch (roundingMode) {
    case 'round':
      // used for Caffe Conv
      return value.round();
    case 'ceil':
      // used for Caffe Pool
      return value.ceil();
    case 'floor':
      return value.floor();
    default:
      throw Exception('Unknown roundingMode ${roundingMode}');
  }
}

bool tupleValuesAreOne(List<int> param // : number|number[]
    ) {
  final l = parseTupleParam(param);
  return l[0] == 1 && l[1] == 1 && l[2] == 1;
}

bool eitherStridesOrDilationsAreOne(
    // : number|number[]
    List<int> strides,
    // : number|number[]
    List<int> dilations) {
  return tupleValuesAreOne(strides) || tupleValuesAreOne(dilations);
}

// /**
//  * Convert Conv2D dataFormat from 'NHWC'|'NCHW' to
//  *    'channelsLast'|'channelsFirst'
//  * @param dataFormat in 'NHWC'|'NCHW' mode
//  * @return dataFormat in 'channelsLast'|'channelsFirst' mode
//  * @throws unknown dataFormat
//  */
// export function convertConv2DDataFormat(dataFormat: 'NHWC'|'NCHW'):
//     'channelsLast'|'channelsFirst' {
//   if (dataFormat === 'NHWC') {
//     return 'channelsLast';
//   } else if (dataFormat === 'NCHW') {
//     return 'channelsFirst';
//   } else {
//     throw Exception(`Unknown dataFormat ${dataFormat}`);
//   }
// }

/**
 * Check validity of pad when using dimRoundingMode.
 * @param opDesc A string of op description
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid` output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @throws unknown padding parameter
 */
void checkPadOnDimRoundingMode(
  String opDesc,
  Object pad,
  // : 'valid'|'same'|number|ExplicitPadding,
  String? dimRoundingMode,
  // ?: 'floor'|'round'|'ceil'
) {
  if (dimRoundingMode != null) {
    if (pad is String) {
      throw Exception('Error in ${opDesc}: pad must be an integer when using ' +
          'dimRoundingMode ${dimRoundingMode} but got pad ${pad}.');
    } else if (pad is num) {
      util.assert_(
          pad is int,
          () =>
              'Error in ${opDesc}: pad must be an integer when using ' +
              'dimRoundingMode ${dimRoundingMode} but got pad ${pad}.');
    } else if (pad is List<List<int>>) {
      pad.forEach((p) {
        p.forEach((v) {
          util.assert_(
              v is int,
              () =>
                  'Error in ${opDesc}: pad must be an integer when using ' +
                  'dimRoundingMode ${dimRoundingMode} but got pad ${v}.');
        });
      });
    } else {
      throw Exception('Error in ${opDesc}: Unknown padding parameter: ${pad}');
    }
  }
}
