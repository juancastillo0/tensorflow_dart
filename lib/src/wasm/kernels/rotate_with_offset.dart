/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// import {KernelConfig, KernelFunc, RotateWithOffset, RotateWithOffsetAttrs, RotateWithOffsetInputs, TensorInfo} from '@tensorflow/tfjs-core';
// import {backend_util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

import 'dart:typed_data';

import '_prelude.dart';
import 'package:tensorflow_wasm/backend_util.dart' as backend_util;

late final Function(List) _wasmRotate;
// : (
//     xId: number, batch: number, imageHeight: number, imageWidth: number,
//     numChannels: number, radians: number, centerX: number, centerY: number,
//     fillBytes: Uint8Array, fillLength: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmRotate = backend.wasm.cwrap(RotateWithOffset, null /* void */, [
    'number', // xId
    'number', // batch
    'number', // imageHeight
    'number', // imageWidth
    'number', // numChannels
    'number', // radians
    'number', // centerX
    'number', // centerY
    'array', // fillBytes
    'number', // fillLength
    'number', // outId
  ]);
}

TensorInfo rotateWithOffset({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final image = inputs['image']!;

  final radians = attrs!['radians'] as double;
  final fillValue = attrs['fillValue'] as List<int>;
  final center = attrs['center'] as List<double>;

  final out = backend.makeOutput(image.shape, image.dtype);
  final imageId = backend.dataIdMap.get(image.dataId)!.id;
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final batch = image.shape[0];
  final imageHeight = image.shape[1];
  final imageWidth = image.shape[2];
  final numChannels = image.shape[3];

  final _center = [center[0] * imageWidth, center[1] * imageHeight];

  final fillIsBlack = fillValue.every((element) => element == 0);
  final fullOpacityValue = 255;

  final fillValues = fillValue is int
      ? [fillValue, fillValue, fillValue, fillIsBlack ? 0 : fullOpacityValue]
          as List<int>
      : [...fillValue, fullOpacityValue];
  final fillBytes = Uint8List.view(Int32List.fromList(fillValues).buffer);

  _wasmRotate([
    imageId,
    batch,
    imageHeight,
    imageWidth,
    numChannels,
    radians,
    _center[0],
    _center[1],
    fillBytes,
    fillValues.length,
    outId
  ]);
  return out;
}

final rotateWithOffsetConfig = KernelConfigG(
  kernelName: RotateWithOffset,
  backendName: 'wasm',
  kernelFunc: rotateWithOffset,
  setupFunc: _setup,
);
