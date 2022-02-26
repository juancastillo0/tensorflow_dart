/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

// import {KernelConfig, KernelFunc, TensorInfo, Transform, TransformAttrs, TransformInputs, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import 'dart:typed_data';

import '_prelude.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

late final Function(List) _wasmTransform;
// : (
//     imageId: number, transformsId: number, isBatchTransform: boolean,
//     batch: number, outHeight: number, outWidth: number, numChannels: number,
//     imageWidth: number, imageHeight: number, strides: Uint8Array,
//     stridesLength: number, interpolationModeId: number, fillModeId: number,
//     fillValue: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmTransform = backend.wasm.cwrap(Transform, null /*void*/, [
    'number', // imageId
    'number', // transformsId
    'bool', // isBatchTransform
    'number', // batch
    'number', // outHeight
    'number', // outWidth
    'number', // numChannels
    'number', // imageWidth
    'number', // imageHeight
    'array', // strides
    'number', // stridesLength
    'number', // interpolationModeId
    'number', // fillModeId
    'number', // fillValue
    'number' // outId
  ]);
}

TensorInfo transform({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final image = inputs['image']!;
  final transforms = inputs['transforms']!;
  final interpolation = attrs!['interpolation'] as String;
  final fillMode = attrs['fillMode'] as String;
  final fillValue = attrs['fillValue'] as double;
  final outputShape = attrs['outputShape'] as List<int>?;

  final batch = image.shape[0];
  final imageHeight = image.shape[1];
  final imageWidth = image.shape[2];
  final numChannels = image.shape[3];

  final outHeight = outputShape?[0] ?? imageHeight;
  final outWidth = outputShape?[1] ?? imageWidth;
  final outShape = [batch, outHeight, outWidth, numChannels];
  final strides = Uint8List.view(
      Int32List.fromList(util.computeStrides(image.shape)).buffer);

  final out = backend.makeOutput(outShape, image.dtype);
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final imageData = backend.dataIdMap.get(image.dataId)!;
  final imageId = imageData.id;

  final transformsData = backend.dataIdMap.get(transforms.dataId)!;
  final transformsId = transformsData.id;

  final interpolationModeId = interpolation == 'nearest' ? 1 : 2;
  final int fillModeId;
  switch (fillMode) {
    case 'constant':
      fillModeId = 1;
      break;
    case 'reflect':
      fillModeId = 2;
      break;
    case 'wrap':
      fillModeId = 3;
      break;
    case 'nearest':
      fillModeId = 4;
      break;
    default:
      fillModeId = 1;
      break;
  }

  _wasmTransform([
    imageId,
    transformsId,
    (transforms.shape[0] > 1),
    batch,
    outHeight,
    outWidth,
    numChannels,
    imageWidth,
    imageHeight,
    strides,
    image.shape.length - 1,
    interpolationModeId,
    fillModeId,
    fillValue,
    outId
  ]);

  return out;
}

final transformConfig = KernelConfigG(
  kernelName: Transform,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: transform,
);
