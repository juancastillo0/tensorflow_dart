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

// import {KernelConfig, KernelFunc, ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

// import {BackendWasm} from '../backend_wasm';

// import {cast} from './Cast';

import '../../util_base.dart' as util;
import '_prelude.dart';
import 'cast.dart';

late final Function(List) _wasmResizeBilinear;
// : (
//     xId: number, batch: number, oldHeight: number, oldWidth: number,
//     numChannels: number, newHeight: number, newWidth: number,
//     alignCorners: number, halfPixelCenters: number, outId: number) => void;

void _setup(BackendWasm backend) {
  _wasmResizeBilinear = backend.wasm.cwrap(ResizeBilinear, null /*void*/, [
    'number', // xId
    'number', // batch
    'number', // oldHeight
    'number', // oldWidth
    'number', // numChannels
    'number', // newHeight
    'number', // newWidth
    'number', // alignCorners
    'number', // halfPixelCenters
    'number' // outId
  ]);
}

TensorInfo resizeBilinear({
  required NamedTensorInfoMap inputs,
  required BackendWasm backend,
  NamedAttrMap? attrs,
}) {
  final images = inputs['images']!;

  final halfPixelCenters = attrs!['halfPixelCenters'] as bool;
  final alignCorners = attrs['alignCorners'] as bool;
  final size = attrs['size'] as List<int>;

  final batch = images.shape[0];
  final oldHeight = images.shape[1];
  final oldWidth = images.shape[2];
  final numChannels = images.shape[3];

  final newHeight = size[0];
  final newWidth = size[1];

  final outShape = [batch, newHeight, newWidth, numChannels];

  var xData = backend.dataIdMap.get(images.dataId)!;
  TensorInfo? castedData;
  if (xData.dtype != 'float32') {
    castedData = cast(
        backend: backend, inputs: {'x': images}, attrs: {'dtype': 'float32'});
    xData = backend.dataIdMap.get(castedData.dataId)!;
  }
  final xId = xData.id;

  final out = backend.makeOutput(outShape, 'float32');
  if (util.sizeFromShape(images.shape) == 0) {
    return out;
  }
  final outId = backend.dataIdMap.get(out.dataId)!.id;

  _wasmResizeBilinear([
    xId,
    batch,
    oldHeight,
    oldWidth,
    numChannels,
    newHeight,
    newWidth,
    alignCorners ? 1 : 0,
    halfPixelCenters ? 1 : 0,
    outId
  ]);

  if (castedData != null) {
    backend.disposeData(castedData.dataId);
  }

  return out;
}

final resizeBilinearConfig = KernelConfigG(
  kernelName: ResizeBilinear,
  backendName: 'wasm',
  setupFunc: _setup,
  kernelFunc: resizeBilinear,
);
