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

// import {complex} from '../ops/complex';
// import {tensor} from '../ops/tensor';
// import {NamedTensor, NamedTensorMap} from '../tensor_types';
// import {TypedArray} from '../types';
// import {sizeFromShape} from '../util';

// import {DTYPE_VALUE_SIZE_MAP, ModelArtifacts, ModelArtifactsInfo, ModelJSON, WeightGroup, WeightsManifestConfig, WeightsManifestEntry} from './types';

import 'dart:convert';
import 'dart:typed_data';

import 'package:tensorflow_wasm/src/io/io.dart';
import 'package:tensorflow_wasm/src/util_base.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';

/** Number of bytes reserved for the length of the string. (32bit integer). */
const NUM_BYTES_STRING_LENGTH = 4;

class EncodedWeights {
  final ByteBuffer data;
  final List<WeightsManifestEntry> specs;

  EncodedWeights({
    required this.data,
    required this.specs,
  });
}

extension ByteBufferSlice on ByteBuffer {
  ByteBuffer slice(int start, [int? end]) {
    final l = asUint8List().sublistRelaxed(start, end) as Uint8List;
    return l.buffer;
  }
}

/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
Future<EncodedWeights> encodeWeights(
  // NamedTensorMap|NamedTensor[]
  NamedTensorMap tensors,
  WeightGroup? group,
) async {
  // TODO(adarob, cais): Support quantization.
  final List<WeightsManifestEntry> specs = [];
  final List<Future<TypedData>> dataPromises = [];

  // final List<String> names= tensors is List<Tensor> ?
  //     tensors.map((tensor) => tensor.name).toList() :
  //     tensors.keys.toList();
  final names = tensors.keys.toList();

  for (int i = 0; i < names.length; ++i) {
    final name = names[i];
    // final t = tensors is List ? tensors[i].tensor : tensors[name];
    final t = tensors[name]!;
    if (t.dtype != 'float32' &&
        t.dtype != 'int32' &&
        t.dtype != 'bool' &&
        t.dtype != 'string' &&
        t.dtype != 'complex64') {
      throw Exception("Unsupported dtype in weight '${name}': ${t.dtype}");
    }
    final spec = WeightsManifestEntry(
      name: name,
      shape: t.shape,
      dtype: t.dtype,
      group: group,
    );
    if (t.dtype == 'string') {
      final utf8bytes = Future(() async {
        final vals = await t.bytes() as List<Uint8List>;
        final totalNumBytes = vals.fold<int>(0, (p, c) => p + c.length) +
            NUM_BYTES_STRING_LENGTH * vals.length;
        final bytes = Uint8List(totalNumBytes);
        int offset = 0;
        for (int i = 0; i < vals.length; i++) {
          final val = vals[i];
          final bytesOfLength =
              Uint8List.view(Uint32List.fromList([val.length]).buffer);

          List.copyRange(bytes, offset,
              bytesOfLength); // bytes.set(bytesOfLength, offset);
          offset += NUM_BYTES_STRING_LENGTH;
          List.copyRange(bytes, offset, val); // bytes.set(val, offset);
          offset += val.length;
        }
        return bytes;
      });
      dataPromises.add(utf8bytes);
    } else {
      dataPromises.add(t.data().then((value) => value as TypedData));
    }
    specs.add(spec);
  }

  final tensorValues = await Future.wait(dataPromises);
  return EncodedWeights(
      data: concatenateTypedArrays(tensorValues), specs: specs);
}

/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param buffer A flat ArrayBuffer carrying the binary values of the tensors
 *   concatenated in the order specified in `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
NamedTensorMap decodeWeights(
  ByteBuffer buffer,
  List<WeightsManifestEntry> specs,
) {
  // TODO(adarob, cais): Support quantization.
  final NamedTensorMap out = {};
  Float32List Function(Uint16List buffer)? float16Decode;
  int offset = 0;
  for (final spec in specs) {
    final name = spec.name;
    final dtype = spec.dtype;
    final shape = spec.shape;
    final size = sizeFromShape(shape);
    List values; // TypedArray|string[]|Uint8Array[]

    final quantization = spec.quantization;
    if (quantization != null) {
      if (quantization.dtype == 'uint8' || quantization.dtype == 'uint16') {
        if (quantization.min == null || quantization.scale == null) {
          throw Exception(
              "Weight ${spec.name} with quantization ${quantization.dtype} " +
                  "doesn't have corresponding metadata min and scale.");
        }
      } else if (quantization.dtype == 'float16') {
        if (dtype != 'float32') {
          throw Exception(
              "Weight ${spec.name} is quantized with ${quantization.dtype} " +
                  "which only supports weights of type float32 not ${dtype}.");
        }
      } else {
        throw Exception("Weight ${spec.name} has unknown " +
            "quantization dtype ${quantization.dtype}. " +
            "Supported quantization dtypes are: " +
            "'uint8', 'uint16', and 'float16'.");
      }
      final quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype]!;
      final byteBuffer =
          buffer.slice(offset, offset + size * quantizationSizeFactor);
      final quantizedArray = (quantization.dtype == 'uint8')
          ? Uint8List.view(byteBuffer)
          : Uint16List.view(byteBuffer);
      if (dtype == 'float32') {
        if (quantization.dtype == 'uint8' || quantization.dtype == 'uint16') {
          values = Float32List(quantizedArray.length);
          for (int i = 0; i < quantizedArray.length; i++) {
            final v = quantizedArray[i];
            values[i] = v * quantization.scale! + quantization.min!;
          }
        } else if (quantization.dtype == 'float16') {
          float16Decode ??= getFloat16Decoder();

          values = float16Decode(quantizedArray as Uint16List);
        } else {
          throw Exception(
              "Unsupported quantization type ${quantization.dtype} " +
                  "for weight type float32.");
        }
      } else if (dtype == 'int32') {
        if (quantization.dtype != 'uint8' && quantization.dtype != 'uint16') {
          throw Exception(
              "Unsupported quantization type ${quantization.dtype} " +
                  "for weight type int32.");
        }
        values = Int32List(quantizedArray.length);
        for (int i = 0; i < quantizedArray.length; i++) {
          final v = quantizedArray[i];
          values[i] = (v * quantization.scale! + quantization.min!).round();
        }
      } else {
        throw Exception("Unsupported dtype in weight '${name}': ${dtype}");
      }
      offset += size * quantizationSizeFactor;
    } else if (dtype == 'string') {
      final size = sizeFromShape(spec.shape);
      values = [];
      for (int i = 0; i < size; i++) {
        final byteLength = Uint32List.view(
            buffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
        offset += NUM_BYTES_STRING_LENGTH;
        final bytes = Uint8List.view(buffer.slice(offset, offset + byteLength));
        (values as List<Uint8List>).add(bytes);
        offset += byteLength;
      }
    } else {
      final dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype]!;
      final byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);

      if (dtype == 'float32') {
        values = Float32List.view(byteBuffer);
      } else if (dtype == 'int32') {
        values = Int32List.view(byteBuffer);
      } else if (dtype == 'bool') {
        values = Uint8List.view(byteBuffer);
      } else if (dtype == 'complex64') {
        values = Float32List.view(byteBuffer);
        final real = Float32List(values.length ~/ 2);
        final image = Float32List(values.length ~/ 2);
        for (int i = 0; i < real.length; i++) {
          real[i] = values[i * 2];
          image[i] = values[i * 2 + 1];
        }
        final realTensor = tensor(real, shape, 'float32');
        final imageTensor = tensor(image, shape, 'float32');
        out[name] = complex(realTensor, imageTensor);
        realTensor.dispose();
        imageTensor.dispose();
      } else {
        throw Exception("Unsupported dtype in weight '${name}': ${dtype}");
      }
      offset += size * dtypeFactor;
    }
    if (dtype != 'complex64') {
      out[name] = tensor(values, shape, dtype);
    }
  }
  return out;
}

/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
ByteBuffer concatenateTypedArrays(List<TypedData> xs) {
  // TODO(adarob, cais): Support quantization.
  if (xs == null) {
    throw Exception("Invalid input value: ${xs}");
  }

  int totalByteLength = 0;

  // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
  // can have a different byte length from that of the `TypedArray` itself,
  // for example, when the `TypedArray` is created from an offset in an
  // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
  // the `TypedArray` in byte length. If an element of `xs` does not show
  // this property, a new `TypedArray` that satisfy this property will be
  // constructed and pushed into `normalizedXs`.
  final List<TypedData> normalizedXs = [];
  xs.forEach((TypedData x) {
    totalByteLength += x.lengthInBytes;
    // tslint:disable:no-any
    if (x.lengthInBytes == x.buffer.lengthInBytes) {
      normalizedXs.add(x);
    } else if (x is Float32List) {
      // TODO: should it be sublistView?
      normalizedXs.add(Float32List.sublistView(x));
    } else if (x is Uint8List) {
      normalizedXs.add(Uint8List.sublistView(x));
    } else if (x is Int32List) {
      normalizedXs.add(Int32List.sublistView(x));
    } else {
      throw Exception("Unsupported TypedArray subtype: ${x.runtimeType}");
    }
    // tslint:enable:no-any
  });

  final y = Uint8List(totalByteLength);
  int offset = 0;
  normalizedXs.forEach((TypedData x) {
    List.copyRange(y, offset,
        Uint8List.view(x.buffer)); // y.set(Uint8List.view(x.buffer), offset);
    offset += x.lengthInBytes;
  });

  return y.buffer;
}

// Use Buffer on Node.js instead of Blob/atob/btoa
// const useNodeBuffer = typeof Buffer != 'undefined' &&
//     (typeof Blob == 'undefined' || typeof atob == 'undefined' ||
//      typeof btoa == 'undefined');

/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
int stringByteLength(String str) {
  return str.length;
  // if (useNodeBuffer) {
  //   return Buffer.byteLength(str);
  // }
  // return new Blob([str]).size;
}

/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
String arrayBufferToBase64String(ByteBuffer buffer) {
  return base64Encode(buffer.asUint8List());
  // if (useNodeBuffer) {
  //   return Buffer.from(buffer).toString('base64');
  // }
  // final buf = Uint8List.view(buffer);
  // String s = '';
  // for (int i = 0, l = buf.length; i < l; i++) {
  //   s += String.fromCharCode(buf[i]);
  // }
  // return btoa(s);
}

/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
ByteBuffer base64StringToArrayBuffer(String str) {
  return base64Decode(str).buffer;
  // if (useNodeBuffer) {
  //   final buf = Buffer.from(str, 'base64');
  //   return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  // }
  // final s = atob(str);
  // final buffer = Uint8List(s.length);
  // for (int i = 0; i < s.length; ++i) {
  //   buffer.set([s.charCodeAt(i)], i);
  // }
  // return buffer.buffer;
}

/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers A number of array buffers to concatenate.
 * @returns Result of concatenating `buffers` in order.
 */
ByteBuffer concatenateArrayBuffers(List<ByteBuffer> buffers) {
  if (buffers.length == 1) {
    return buffers[0];
  }

  int totalByteLength = 0;
  buffers.forEach((buffer) {
    totalByteLength += buffer.lengthInBytes;
  });

  final temp = Uint8List(totalByteLength);
  int offset = 0;
  buffers.forEach((buffer) {
    List.copyRange(temp, offset,
        Uint8List.view(buffer)); // temp.set(Uint8List(buffer), offset);
    offset += buffer.lengthInBytes;
  });
  return temp.buffer;
}

/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
String basename(String path) {
  const SEPARATOR = '/';
  path = path.trim();
  while (path.endsWith(SEPARATOR)) {
    path = path.substring(0, path.length - 1);
  }
  final items = path.split(SEPARATOR);
  return items[items.length - 1];
}

/**
 * Create `ModelJSON` from `ModelArtifacts`.
 *
 * @param artifacts Model artifacts, describing the model and its weights.
 * @param manifest Weight manifest, describing where the weights of the
 *     `ModelArtifacts` are stored, and some metadata about them.
 * @returns Object representing the `model.json` file describing the model
 *     artifacts and weights
 */
ModelJSON getModelJSONForModelArtifacts(
    ModelArtifacts artifacts, WeightsManifestConfig manifest) {
  final result = ModelJSON(
    modelTopology: artifacts.modelTopology as Map,
    format: artifacts.format,
    generatedBy: artifacts.generatedBy,
    convertedBy: artifacts.convertedBy,
    weightsManifest: manifest,
    signature: artifacts.signature,
    userDefinedMetadata: artifacts.userDefinedMetadata,
    modelInitializer: artifacts.modelInitializer,
    trainingConfig: artifacts.trainingConfig,
  );
  return result;
}

/**
 * Create `ModelArtifacts` from a JSON file.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param loadWeights Function that takes the JSON file's weights manifest,
 *     reads weights from the listed path(s), and returns a Promise of the
 *     weight manifest entries along with the weights data.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
Future<ModelArtifacts> getModelArtifactsForJSON(
    ModelJSON modelJSON,
    Future<
                EncodedWeights
                // [ /* weightSpecs */ WeightsManifestEntry[], /* weightData */ ArrayBuffer]
                >
            Function(WeightsManifestConfig weightsManifest)
        loadWeights) async {
  final weights = await loadWeights(modelJSON.weightsManifest);
  final modelArtifacts = ModelArtifacts(
    modelTopology: modelJSON.modelTopology,
    format: modelJSON.format,
    generatedBy: modelJSON.generatedBy,
    convertedBy: modelJSON.convertedBy,
    trainingConfig: modelJSON.trainingConfig,
    weightSpecs: weights.specs,
    weightData: weights.data,
    signature: modelJSON.signature,
    userDefinedMetadata: modelJSON.userDefinedMetadata,
    modelInitializer: modelJSON.modelInitializer,
  );

  return modelArtifacts;
}

/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
ModelArtifactsInfo getModelArtifactsInfoForJSON(ModelArtifacts modelArtifacts) {
  if (modelArtifacts.modelTopology is ByteBuffer) {
    throw Exception('Expected JSON model topology, received ArrayBuffer.');
  }

  return ModelArtifactsInfo(
    dateSaved: DateTime.now(),
    modelTopologyType: ModelTopologyType.JSON,
    modelTopologyBytes: modelArtifacts.modelTopology == null
        ? 0
        : stringByteLength(jsonEncode(modelArtifacts.modelTopology)),
    weightSpecsBytes: modelArtifacts.weightSpecs == null
        ? 0
        : stringByteLength(jsonEncode(modelArtifacts.weightSpecs)),
    weightDataBytes: modelArtifacts.weightData == null
        ? 0
        : modelArtifacts.weightData!.lengthInBytes,
  );
}

/**
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
Uint32List _computeFloat16MantisaTable() {
  int convertMantissa(int i) {
    int m = i << 13;
    int e = 0;

    while ((m & 0x00800000) == 0) {
      e -= 0x00800000;
      m <<= 1;
    }
    m &= ~0x00800000;
    e += 0x38800000;

    return m | e;
  }

  ;

  final mantisaTable = Uint32List(2048);

  mantisaTable[0] = 0;
  for (int i = 1; i < 1024; i++) {
    mantisaTable[i] = convertMantissa(i);
  }
  for (int i = 1024; i < 2048; i++) {
    mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
  }

  return mantisaTable;
}

/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
Uint32List _computeFloat16ExponentTable() {
  final exponentTable = Uint32List(64);

  exponentTable[0] = 0;
  exponentTable[31] = 0x47800000;
  exponentTable[32] = 0x80000000;
  exponentTable[63] = 0xc7800000;
  for (int i = 1; i < 31; i++) {
    exponentTable[i] = i << 23;
  }
  for (int i = 33; i < 63; i++) {
    exponentTable[i] = 0x80000000 + ((i - 32) << 23);
  }

  return exponentTable;
}

/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
Uint32List _computeFloat16OffsetTable() {
  final offsetTable = Uint32List(64);

  for (int i = 0; i < 64; i++) {
    offsetTable[i] = 1024;
  }
  offsetTable[0] = offsetTable[32] = 0;

  return offsetTable;
}

/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
Float32List Function(Uint16List buffer) getFloat16Decoder() {
  // Algorithm is based off of
  // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf

  // Cache lookup tables
  final mantisaTable = _computeFloat16MantisaTable();
  final exponentTable = _computeFloat16ExponentTable();
  final offsetTable = _computeFloat16OffsetTable();

  return (Uint16List quantizedArray) {
    final buffer = ByteData(4 * quantizedArray.length);
    final bufferUint32View = Uint32List.sublistView(buffer);
    for (int index = 0; index < quantizedArray.length; index++) {
      final float16Bits = quantizedArray[index];
      final float32Bits =
          mantisaTable[offsetTable[float16Bits >> 10] + (float16Bits & 0x3ff)] +
              exponentTable[float16Bits >> 10];
      bufferUint32View[index] = float32Bits;
    }
    return Float32List.sublistView(buffer);
  };
}
