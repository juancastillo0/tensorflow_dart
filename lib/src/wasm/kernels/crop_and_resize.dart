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

// import {CropAndResize, CropAndResizeAttrs, CropAndResizeInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {cast} from './Cast';

import 'dart:typed_data';

import '_prelude.dart';
import 'cast.dart';

// Must match enum in CropAndResize.cc
enum InterpolationMethod {
  bilinear,
  nearest,
}

late final Function(List) _wasmCropAndResize;
// : (
//     imagesId: number, boxesId: number, boxIndId: number, numBoxes: number,
//     imagesShape: Uint8Array, cropHeight: number, cropWidth: number,
//     method: number, extrapolationValue: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmCropAndResize = backend.wasm.cwrap(CropAndResize, null /*void*/, [
    'number', // imagesId
    'number', // boxesId
    'number', // boxIndId
    'number', // numBoxes
    'array', // images shape
    'number', // cropHeight
    'number', // cropWidth
    'number', // method
    'number', // extrapolation value
    'number' // out id
  ]);
}

TensorInfo cropAndResize({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final image = inputs['image']!;
  final boxes = inputs['boxes']!;
  final boxInd = inputs['boxInd']!;

  final cropSize = attrs!['cropSize'] as List<int>;
  final extrapolationValue = attrs['extrapolationValue'] as double;
  final method = attrs['method'] as String;

  final numBoxes = boxes.shape[0];

  final cropHeight = cropSize[0];
  final cropWidth = cropSize[1];
  final outShape = [numBoxes, cropHeight, cropWidth, image.shape[3]];

  var imagesData = backend.dataIdMap.get(image.dataId)!;
  TensorInfo? castedData;
  if (image.dtype != 'float32') {
    castedData = cast(
        backend: backend, inputs: {'x': image}, attrs: {'dtype': 'float32'});
    imagesData = backend.dataIdMap.get(castedData.dataId)!;
  }

  final imagesId = imagesData.id;
  final boxesId = backend.dataIdMap.get(boxes.dataId)!.id;
  final boxIndId = backend.dataIdMap.get(boxInd.dataId)!.id;

  final out = backend.makeOutput(outShape, 'float32');
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  final imagesShapeBytes =
      Uint8List.view(Int32List.fromList(image.shape).buffer);

  _wasmCropAndResize([
    imagesId,
    boxesId,
    boxIndId,
    numBoxes,
    imagesShapeBytes,
    cropHeight,
    cropWidth,
    InterpolationMethod.values.byName(method).index,
    extrapolationValue,
    outId
  ]);

  if (castedData != null) {
    backend.disposeData(castedData.dataId);
  }

  return out;
}

final cropAndResizeConfig = KernelConfigG(
  kernelName: CropAndResize,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: cropAndResize,
);
