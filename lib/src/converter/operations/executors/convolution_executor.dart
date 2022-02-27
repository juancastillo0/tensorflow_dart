/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// import {Rank, Tensor, Tensor3D, Tensor4D, Tensor5D} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpExecutor, Node} from '../types';

// import {getPadding, getParamValue} from './utils';

import 'package:tensorflow_wasm/backend_util.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

class _ConvParams {
  final List<int> stride;
  final Object pad;
  final String dataFormat;
  final List<int> dilations;
  final Tensor? biasArg;
  final Tensor preluArg;
  final String activationFunc;
  final double leakyreluAlpha;

  _ConvParams({
    required this.stride,
    required this.pad,
    required this.dataFormat,
    required this.dilations,
    required this.biasArg,
    required this.preluArg,
    required this.activationFunc,
    required this.leakyreluAlpha,
  });
}

_ConvParams fusedConvAndDepthWiseParams(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  final params =
      (getParamValue('fusedOps', node, tensorMap, context) as List<String>);
  final extraOp = params[0];
  final activationFunc = params[1];

  final isBiasAdd = extraOp == 'biasadd';
  final noBiasAdd = !isBiasAdd;
  final isPrelu = activationFunc == 'prelu';
  final isBatchNorm = extraOp == 'fusedbatchnorm';

  final numArgs = (getParamValue('numArgs', node, tensorMap, context) as int);
  if (isBiasAdd) {
    if (isPrelu && numArgs != 2) {
      throw Exception(
          'FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu ' +
              'must have two extra arguments: bias and alpha.');
    }
    if (!isPrelu && isBiasAdd && numArgs != 1) {
      throw Exception(
          'FusedConv2d and DepthwiseConv2d with BiasAdd must have ' +
              'one extra argument: bias.');
    }
  }
  if (isBatchNorm) {
    throw Exception(
        'FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported');
  }
  final stride =
      getParamValue('strides', node, tensorMap, context) as List<int>;
  final pad = getPadding(node, tensorMap, context)!;
  final dataFormat =
      (getParamValue('dataFormat', node, tensorMap, context) as String)
          .toUpperCase();
  final dilations =
      getParamValue('dilations', node, tensorMap, context) as List<int>;
  final _a = getParamValue('args', node, tensorMap, context) as List<Tensor>;

  Tensor? biasArg = _a[0];
  Tensor preluArg = _a[1];
  if (noBiasAdd) {
    preluArg = biasArg;
    biasArg = null;
  }
  final leakyreluAlpha =
      getParamValue('leakyreluAlpha', node, tensorMap, context) as double;

  return _ConvParams(
      stride: stride,
      pad: pad,
      dataFormat: dataFormat,
      dilations: dilations,
      biasArg: biasArg,
      preluArg: preluArg,
      activationFunc: activationFunc,
      leakyreluAlpha: leakyreluAlpha);
}

List<Tensor> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  switch (node.op) {
    case 'Conv1D':
      {
        final stride = getParamValue('stride', node, tensorMap, context) as int;
        final pad = getParamValue('pad', node, tensorMap, context)!;
        final dataFormat =
            (getParamValue('dataFormat', node, tensorMap, context) as String)
                .toUpperCase();
        final dilation =
            getParamValue('dilation', node, tensorMap, context) as int;
        return [
          tfOps.conv1d(getParamValue('x', node, tensorMap, context) as Tensor3D,
              getParamValue('filter', node, tensorMap, context) as Tensor3D,
              stride: stride,
              pad: pad, // as 'valid' | 'same',
              dataFormat: dataFormat, //  as 'NWC' | 'NCW',
              dilation: dilation)
        ];
      }
    case 'Conv2D':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getPadding(node, tensorMap, context)!;
        final dataFormat =
            (getParamValue('dataFormat', node, tensorMap, context) as String)
                .toUpperCase();
        final dilations =
            getParamValue('dilations', node, tensorMap, context) as List<int>;
        return [
          tfOps.conv2d(
            getParamValue('x', node, tensorMap, context)
                as Tensor3D, //  |Tensor4D,
            getParamValue('filter', node, tensorMap, context) as Tensor4D,
            strides: [stride[1], stride[2]],
            pad: pad, // as 'valid' | 'same',
            dataFormat: dataFormat, // as 'NHWC' | 'NCHW',
            dilations: [dilations[1], dilations[2]],
          )
        ];
      }
    case '_FusedConv2D':
      {
        final p = fusedConvAndDepthWiseParams(node, tensorMap, context);

        return [
          tfOps.fused.conv2d(
            x: getParamValue('x', node, tensorMap, context)
                as Tensor3D, // |  Tensor4D,
            filter:
                getParamValue('filter', node, tensorMap, context) as Tensor4D,
            strides: [p.stride[1], p.stride[2]],
            pad: p.pad, // as 'valid' | 'same',
            dataFormat: p.dataFormat, // as 'NHWC' | 'NCHW',
            dilations: [p.dilations[1], p.dilations[2]],
            bias: p.biasArg,
            activation: Activation.values.byName(p.activationFunc),
            preluActivationWeights: p.preluArg,
            leakyreluAlpha: p.leakyreluAlpha,
          )
        ];
      }

    case 'FusedDepthwiseConv2dNative':
      {
        final p = fusedConvAndDepthWiseParams(node, tensorMap, context);

        return [
          tfOps.fused.depthwiseConv2d(
              x: getParamValue('x', node, tensorMap, context)
                  as Tensor3D, // |Tensor4D,
              filter:
                  getParamValue('filter', node, tensorMap, context) as Tensor4D,
              strides: [p.stride[1], p.stride[2]],
              pad: p.pad, // as 'valid' | 'same',
              dataFormat: p.dataFormat, // as 'NHWC' | 'NCHW',
              dilations: [p.dilations[1], p.dilations[2]],
              bias: p.biasArg,
              activation: Activation.values.byName(p.activationFunc),
              preluActivationWeights: p.preluArg,
              leakyreluAlpha: p.leakyreluAlpha)
        ];
      }
    case 'Conv2DBackpropInput':
    case 'Conv2dTranspose':
      {
        final shape =
            getParamValue('outputShape', node, tensorMap, context) as List<int>;
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getPadding(node, tensorMap, context)!;
        return [
          tfOps.conv2dTranspose(
            getParamValue('x', node, tensorMap, context)
                as Tensor3D, // | Tensor4D,
            getParamValue('filter', node, tensorMap, context) as Tensor4D,
            outputShape: shape, strides: [stride[1], stride[2]],
            pad: pad, //  as 'valid' | 'same'
          )
        ];
      }
    case 'DepthwiseConv2dNative':
    case 'DepthwiseConv2d':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getPadding(node, tensorMap, context)!;
        final dilations =
            getParamValue('dilations', node, tensorMap, context) as List<int>;
        final dataFormat =
            (getParamValue('dataFormat', node, tensorMap, context) as String)
                .toUpperCase();

        return [
          tfOps.depthwiseConv2d(
              getParamValue('input', node, tensorMap, context)
                  as Tensor3D, // | Tensor4D,
              getParamValue('filter', node, tensorMap, context) as Tensor4D,
              strides: [stride[1], stride[2]],
              pad: pad, // as 'valid' | 'same',
              dataFormat: dataFormat, // as 'NHWC' | 'NCHW',
              dilations: [dilations[1], dilations[2]])
        ];
      }
    case 'Conv3D':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getParamValue('pad', node, tensorMap, context);
        final dataFormat =
            (getParamValue('dataFormat', node, tensorMap, context) as String)
                .toUpperCase();
        final dilations =
            getParamValue('dilations', node, tensorMap, context) as List<int>;
        return [
          tfOps.conv3d(
              getParamValue('x', node, tensorMap, context)
                  as Tensor4D, // | Tensor<Rank.R5>,
              getParamValue('filter', node, tensorMap, context) as Tensor5D,
              [stride[1], stride[2], stride[3]],
              pad, //as 'valid' | 'same',
              dataFormat, // as 'NDHWC' | 'NCDHW',
              [dilations[1], dilations[2], dilations[3]])
        ];
      }
    case 'AvgPool':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getParamValue('pad', node, tensorMap, context)!;
        final kernelSize =
            getParamValue('kernelSize', node, tensorMap, context) as List<int>;

        return [
          tfOps.avgPool(
            getParamValue('x', node, tensorMap, context)
                as Tensor3D, // | Tensor4D,
            filterSize: [kernelSize[1], kernelSize[2]],
            strides: [stride[1], stride[2]],
            pad: pad, // as 'valid' | 'same'
          )
        ];
      }
    case 'MaxPool':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getParamValue('pad', node, tensorMap, context)!;
        final kernelSize =
            getParamValue('kernelSize', node, tensorMap, context) as List<int>;

        return [
          tfOps.maxPool(
            getParamValue('x', node, tensorMap, context)
                as Tensor3D, //| Tensor4D,
            filterSize: [kernelSize[1], kernelSize[2]],
            strides: [stride[1], stride[2]],
            pad: pad, // as 'valid' | 'same'
          )
        ];
      }
    case 'MaxPoolWithArgmax':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getParamValue('pad', node, tensorMap, context);
        final kernelSize =
            getParamValue('kernelSize', node, tensorMap, context) as List<int>;
        final includeBatchInIndex =
            getParamValue('includeBatchInIndex', node, tensorMap, context)
                as bool;
        final p = tfOps.maxPoolWithArgmax(
            getParamValue('x', node, tensorMap, context) as Tensor4D,
            [kernelSize[1], kernelSize[2]],
            [stride[1], stride[2]],
            pad //as 'valid' | 'same'
            ,
            includeBatchInIndex);

        return [p.result, p.indexes];
      }
    case 'AvgPool3D':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getParamValue('pad', node, tensorMap, context);
        final kernelSize =
            getParamValue('kernelSize', node, tensorMap, context) as List<int>;

        return [
          tfOps.avgPool3d(
              getParamValue('x', node, tensorMap, context) as Tensor5D,
              [kernelSize[1], kernelSize[2], kernelSize[3]],
              [stride[1], stride[2], stride[3]],
              pad // as 'valid' | 'same'
              )
        ];
      }

    case 'MaxPool3D':
      {
        final stride =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getParamValue('pad', node, tensorMap, context);
        final kernelSize =
            getParamValue('kernelSize', node, tensorMap, context) as List<int>;

        return [
          tfOps.maxPool3d(
              getParamValue('x', node, tensorMap, context) as Tensor5D,
              [kernelSize[1], kernelSize[2], kernelSize[3]],
              [stride[1], stride[2], stride[3]],
              pad // as 'valid' | 'same'
              )
        ];
      }

    case 'Dilation2D':
      {
        final strides =
            getParamValue('strides', node, tensorMap, context) as List<int>;
        final pad = getParamValue('pad', node, tensorMap, context);
        final dilations =
            getParamValue('dilations', node, tensorMap, context) as List<int>;

        // strides: [1, stride_height, stride_width, 1].
        final strideHeight = strides[1];
        final strideWidth = strides[2];

        // dilations: [1, dilation_height, dilation_width, 1].
        final dilationHeight = dilations[1];
        final dilationWidth = dilations[2];

        return [
          tfOps.dilation2d(
              getParamValue('x', node, tensorMap, context)
                  as Tensor3D, //| Tensor4D,
              getParamValue('filter', node, tensorMap, context) as Tensor3D,
              [strideHeight, strideWidth],
              pad, // as 'valid' | 'same',
              [dilationHeight, dilationWidth],
              'NHWC' /* dataFormat */)
        ];
      }

    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'convolution';
