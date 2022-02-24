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

// import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
// // tslint:disable-next-line: no-imports-from-dist
// import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../../data/types';
// import {ExecutionContext} from '../../executor/execution_context';
// import {InternalOpExecutor, Node} from '../types';

// import {getParamValue} from './utils';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tfOps;
import '_prelude.dart';

List<Tensor> executeOp(
  Node node,
  NamedTensorsMap tensorMap,
  ExecutionContext context,
) {
  switch (node.op) {
    case 'ResizeBilinear':
      {
        final images =
            getParamValue('images', node, tensorMap, context) as Tensor;
        final size =
            getParamValue('size', node, tensorMap, context) as List<int>;
        final alignCorners =
            getParamValue('alignCorners', node, tensorMap, context) as bool;
        final halfPixelCenters =
            getParamValue('halfPixelCenters', node, tensorMap, context) as bool;
        return [
          tfOps.image.resizeBilinear(
              images as Tensor3D // | Tensor4D
              ,
              [size[0], size[1]],
              alignCorners: alignCorners,
              halfPixelCenters: halfPixelCenters)
        ];
      }
    case 'ResizeNearestNeighbor':
      {
        final images =
            getParamValue('images', node, tensorMap, context) as Tensor;
        final size =
            getParamValue('size', node, tensorMap, context) as List<int>;
        final alignCorners =
            getParamValue('alignCorners', node, tensorMap, context) as bool;
        final halfPixelCenters =
            getParamValue('halfPixelCenters', node, tensorMap, context) as bool;
        return [
          tfOps.image.resizeNearestNeighbor(
              images as Tensor3D // | Tensor4D
              ,
              [size[0], size[1]],
              alignCorners: alignCorners,
              halfPixelCenters: halfPixelCenters)
        ];
      }
    case 'CropAndResize':
      {
        final image =
            getParamValue('image', node, tensorMap, context) as Tensor;
        final boxes =
            getParamValue('boxes', node, tensorMap, context) as Tensor;
        final boxInd =
            getParamValue('boxInd', node, tensorMap, context) as Tensor;
        final cropSize =
            getParamValue('cropSize', node, tensorMap, context) as List<int>;
        final method =
            getParamValue('method', node, tensorMap, context) as String;
        final extrapolationValue =
            getParamValue('extrapolationValue', node, tensorMap, context)
                as double;
        return [
          tfOps.image.cropAndResize(image as Tensor4D, boxes as Tensor2D,
              boxInd as Tensor1D, cropSize, // as [number, number],
              method: method, //  as 'bilinear' | 'nearest',
              extrapolationValue: extrapolationValue)
        ];
      }
    default:
      throw StateError('Node type ${node.op} is not implemented');
  }
}

const CATEGORY = 'image';
