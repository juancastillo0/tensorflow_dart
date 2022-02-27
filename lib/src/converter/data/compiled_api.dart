// ignore_for_file: camel_case_types, constant_identifier_names

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
 *
 * =============================================================================
 */

/* tslint:disable */

import 'dart:typed_data';

/** Properties of an Any. */
abstract class IAny {
  /** Any typeUrl */
  String? typeUrl;

  /** Any value */
  Uint8List? value;
}

/** DataType enum. */
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID, // = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT, // = 1,
  DT_DOUBLE, // = 2,
  DT_INT32, // = 3,
  DT_UINT8, // = 4,
  DT_INT16, // = 5,
  DT_INT8, // = 6,
  DT_STRING, // = 7,
  DT_COMPLEX64, // = 8,  // Single-precision complex
  DT_INT64, // = 9,
  DT_BOOL, // = 10,
  DT_QINT8, // = 11,     // Quantized int8
  DT_QUINT8, // = 12,    // Quantized uint8
  DT_QINT32, // = 13,    // Quantized int32
  DT_BFLOAT16, // = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16, // = 15,    // Quantized int16
  DT_QUINT16, // = 16,   // Quantized uint16
  DT_UINT16, // = 17,
  DT_COMPLEX128, // = 18,  // Double-precision complex
  DT_HALF, // = 19,
  DT_RESOURCE, // = 20,
  DT_VARIANT, // = 21,  // Arbitrary C++ data types
  DT_UINT32, // = 22,
  DT_UINT64, // = 23,

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  DT_FLOAT_REF, // = 101,
  DT_DOUBLE_REF, // = 102,
  DT_INT32_REF, // = 103,
  DT_UINT8_REF, // = 104,
  DT_INT16_REF, // = 105,
  DT_INT8_REF, // = 106,
  DT_STRING_REF, // = 107,
  DT_COMPLEX64_REF, // = 108,
  DT_INT64_REF, // = 109,
  DT_BOOL_REF, // = 110,
  DT_QINT8_REF, // = 111,
  DT_QUINT8_REF, // = 112,
  DT_QINT32_REF, // = 113,
  DT_BFLOAT16_REF, // = 114,
  DT_QINT16_REF, // = 115,
  DT_QUINT16_REF, // = 116,
  DT_UINT16_REF, // = 117,
  DT_COMPLEX128_REF, // = 118,
  DT_HALF_REF, // = 119,
  DT_RESOURCE_REF, // = 120,
  DT_VARIANT_REF, // = 121,
  DT_UINT32_REF, // = 122,
  DT_UINT64_REF, // = 123,
}

/** Properties of a TensorShape. */
class ITensorShape {
  /** TensorShape dim */
  final List<TensorShape_IDim>? dim;

  /** TensorShape unknownRank */
  final bool? unknownRank;

  ITensorShape({
    this.dim,
    this.unknownRank,
  });

  factory ITensorShape.fromJson(Map<String, Object?> json) => ITensorShape(
        dim: json['dim'] == null
            ? null
            : (json['dim'] as List)
                .map((e) => TensorShape_IDim.fromJson(e))
                .toList(),
        unknownRank: json['unknownRank'] as bool?,
      );
}

// class TensorShape {
/** Properties of a Dim. */
class TensorShape_IDim {
  /** Dim size */
  final Object? size; // number|string

  /** Dim name */
  final String? name;

  TensorShape_IDim({
    this.size,
    this.name,
  });

  factory TensorShape_IDim.fromJson(Map<String, Object?> json) =>
      TensorShape_IDim(
        name: json['name'] as String?,
        size: json['size'],
      );
}
// }

/** Properties of a Tensor. */
abstract class ITensor {
  /** Tensor dtype */
  DataType? dtype;

  /** Tensor tensorShape */
  ITensorShape? tensorShape;

  /** Tensor versionNumber */
  int? versionNumber;

  /** Tensor tensorContent */
  Uint8List? tensorContent;

  /** Tensor floatVal */
  List<int>? floatVal;

  /** Tensor doubleVal */
  List<int>? doubleVal;

  /** Tensor intVal */
  List<int>? intVal;

  /** Tensor stringVal */
  List<Uint8List>? stringVal;

  /** Tensor scomplexVal */
  List<int>? scomplexVal;

  /** Tensor int64Val */
  List<Object?>? int64Val; // (number | string)[]

  /** Tensor boolVal */
  List<bool>? boolVal;

  /** Tensor uint32Val */
  List<int>? uint32Val;

  /** Tensor uint64Val */
  List<Object?>? uint64Val; // (number | string)[]
}

/** Properties of an AttrValue. */
abstract class IAttrValue {
  /** AttrValue list */
  AttrValue_IListValue? list;

  /** AttrValue s */
  String? s;

  /** AttrValue i */
  Object? i; // number|string

  /** AttrValue f */
  int? f;

  /** AttrValue b */
  bool? b;

  /** AttrValue type */
  DataType? type;

  /** AttrValue shape */
  ITensorShape? shape;

  /** AttrValue tensor */
  ITensor? tensor;

  /** AttrValue placeholder */
  String? placeholder;

  /** AttrValue func */
  INameAttrList? func;
}

// class AttrValue {
/** Properties of a ListValue. */
abstract class AttrValue_IListValue {
  /** ListValue s */
  List<String>? s;

  /** ListValue i */
  List<Object>? i; // (number | string)

  /** ListValue f */
  List<int>? f;

  /** ListValue b */
  List<bool>? b;

  /** ListValue type */
  List<DataType>? type;

  /** ListValue shape */
  List<ITensorShape>? shape;

  /** ListValue tensor */
  List<ITensor>? tensor;

  /** ListValue func */
  List<INameAttrList>? func;
}
// }

/** Properties of a NameAttrList. */
abstract class INameAttrList {
  /** NameAttrList name */
  String? name;

  /** NameAttrList attr */
  Map<String, IAttrValue>? attr;
}

/** Properties of a NodeDef. */
abstract class INodeDef {
  /** NodeDef name */
  String? name;

  /** NodeDef op */
  String? op;

  /** NodeDef input */
  List<String>? input;

  /** NodeDef device */
  String? device;

  /** NodeDef attr */
  Map<String, IAttrValue>? attr;
}

/** Properties of a VersionDef. */
abstract class IVersionDef {
  /** VersionDef producer */
  int? producer;

  /** VersionDef minConsumer */
  int? minConsumer;

  /** VersionDef badConsumers */
  List<int>? badConsumers;
}

/** Properties of a GraphDef. */
abstract class IGraphDef {
  /** GraphDef node */
  List<INodeDef>? node;

  /** GraphDef versions */
  IVersionDef? versions;

  /** GraphDef library */
  IFunctionDefLibrary? library;
}

/** Properties of a CollectionDef. */
abstract class ICollectionDef {
  /** CollectionDef nodeList */
  CollectionDef_INodeList? nodeList;

  /** CollectionDef bytesList */
  CollectionDef_IBytesList? bytesList;

  /** CollectionDef int64List */
  CollectionDef_IInt64List? int64List;

  /** CollectionDef floatList */
  CollectionDef_IFloatList? floatList;

  /** CollectionDef anyList */
  CollectionDef_IAnyList? anyList;
}

// class CollectionDef {
/** Properties of a NodeList. */
abstract class CollectionDef_INodeList {
  /** NodeList value */
  List<String>? value;
}

/** Properties of a BytesList. */
abstract class CollectionDef_IBytesList {
  /** BytesList value */
  List<Uint8List>? value;
}

/** Properties of an Int64List. */
abstract class CollectionDef_IInt64List {
  /** Int64List value */
  List<Object>? value; // number | string
}

/** Properties of a FloatList. */
abstract class CollectionDef_IFloatList {
  /** FloatList value */
  List<int>? value;
}

/** Properties of an AnyList. */
abstract class CollectionDef_IAnyList {
  /** AnyList value */
  List<IAny>? value;
}
// }

/** Properties of a SaverDef. */
abstract class ISaverDef {
  /** SaverDef filenameTensorName */
  String? filenameTensorName;

  /** SaverDef saveTensorName */
  String? saveTensorName;

  /** SaverDef restoreOpName */
  String? restoreOpName;

  /** SaverDef maxToKeep */
  int? maxToKeep;

  /** SaverDef sharded */
  bool? sharded;

  /** SaverDef keepCheckpointEveryNHours */
  int? keepCheckpointEveryNHours;

  /** SaverDef version */
  SaverDef_CheckpointFormatVersion? version;
}

// class SaverDef {
/** CheckpointFormatVersion enum. */
enum SaverDef_CheckpointFormatVersion { LEGACY, V1, V2 }
// }

/** Properties of a TensorInfo. */
class ITensorInfo {
  /** TensorInfo name */
  final String? name;

  /** TensorInfo cooSparse */
  final TensorInfo_ICooSparse? cooSparse;

  /** TensorInfo dtype */
  final DataType? dtype;

  /** TensorInfo tensorShape */
  final ITensorShape? tensorShape;

  ITensorInfo({
    this.name,
    this.cooSparse,
    this.dtype,
    this.tensorShape,
  });

  factory ITensorInfo.fromJson(Map<String, Object?> json) => ITensorInfo(
        name: json['name'] as String?,
        cooSparse: json['cooSparse'] == null
            ? null
            : TensorInfo_ICooSparse.fromMap(
                json['cooSparse'] as Map<String, Object?>),
        dtype: json['dtype'] == null
            ? null
            : DataType.values[json['dtype'] as int],
        tensorShape: json['tensorShape'] == null
            ? null
            : ITensorShape.fromJson(
                json['tensorShape'] as Map<String, Object?>),
      );
}

// class TensorInfo {
/** Properties of a CooSparse. */
class TensorInfo_ICooSparse {
  /** CooSparse valuesTensorName */
  final String? valuesTensorName;

  /** CooSparse indicesTensorName */
  final String? indicesTensorName;

  /** CooSparse denseShapeTensorName */
  final String? denseShapeTensorName;

  TensorInfo_ICooSparse({
    this.valuesTensorName,
    this.indicesTensorName,
    this.denseShapeTensorName,
  });

  Map<String, dynamic> toMap() {
    return {
      'valuesTensorName': valuesTensorName,
      'indicesTensorName': indicesTensorName,
      'denseShapeTensorName': denseShapeTensorName,
    };
  }

  factory TensorInfo_ICooSparse.fromMap(Map<String, dynamic> map) {
    return TensorInfo_ICooSparse(
      valuesTensorName: map['valuesTensorName'],
      indicesTensorName: map['indicesTensorName'],
      denseShapeTensorName: map['denseShapeTensorName'],
    );
  }
}
// }

/** Properties of a SignatureDef. */
class ISignatureDef {
  /** SignatureDef inputs */
  final Map<String, ITensorInfo>? inputs;

  /** SignatureDef outputs */
  final Map<String, ITensorInfo>? outputs;

  /** SignatureDef methodName */
  final String? methodName;

  const ISignatureDef({
    this.inputs,
    this.outputs,
    this.methodName,
  });

  factory ISignatureDef.fromJson(Map<String, Object?> json) => ISignatureDef(
        methodName: json['methodName'] as String?,
        inputs: (json['inputs'] as Map<String, Object?>).map((key, value) =>
            MapEntry(key, ITensorInfo.fromJson(value as Map<String, Object?>))),
        outputs: (json['outputs'] as Map<String, Object?>).map((key, value) =>
            MapEntry(key, ITensorInfo.fromJson(value as Map<String, Object?>))),
      );
}

/** Properties of an AssetFileDef. */
abstract class IAssetFileDef {
  /** AssetFileDef tensorInfo */
  ITensorInfo? tensorInfo;

  /** AssetFileDef filename */
  String? filename;
}

/** Properties of an OpDef. */
abstract class IOpDef {
  /** OpDef name */
  String? name;

  /** OpDef inputArg */
  List<OpDef_IArgDef>? inputArg;

  /** OpDef outputArg */
  List<OpDef_IArgDef>? outputArg;

  /** OpDef attr */
  List<OpDef_IAttrDef>? attr;

  /** OpDef deprecation */
  OpDef_IOpDeprecation? deprecation;

  /** OpDef summary */
  String? summary;

  /** OpDef description */
  String? description;

  /** OpDef isCommutative */
  bool? isCommutative;

  /** OpDef isAggregate */
  bool? isAggregate;

  /** OpDef isStateful */
  bool? isStateful;

  /** OpDef allowsUninitializedInput */
  bool? allowsUninitializedInput;
}

// class OpDef {
/** Properties of an ArgDef. */
abstract class OpDef_IArgDef {
  /** ArgDef name */
  String? name;

  /** ArgDef description */
  String? description;

  /** ArgDef type */
  DataType? type;

  /** ArgDef typeAttr */
  String? typeAttr;

  /** ArgDef numberAttr */
  String? numberAttr;

  /** ArgDef typeListAttr */
  String? typeListAttr;

  /** ArgDef isRef */
  bool? isRef;
}

/** Properties of an AttrDef. */
abstract class OpDef_IAttrDef {
  /** AttrDef name */
  String? name;

  /** AttrDef type */
  String? type;

  /** AttrDef defaultValue */
  IAttrValue? defaultValue;

  /** AttrDef description */
  String? description;

  /** AttrDef hasMinimum */
  bool? hasMinimum;

  /** AttrDef minimum */
  Object? minimum; // number|String

  /** AttrDef allowedValues */
  IAttrValue? allowedValues;
}

/** Properties of an OpDeprecation. */
abstract class OpDef_IOpDeprecation {
  /** OpDeprecation version */
  int? version;

  /** OpDeprecation explanation */
  String? explanation;
}
// }

/** Properties of an OpList. */
abstract class IOpList {
  /** OpList op */
  List<IOpDef>? op;
}

/** Properties of a MetaGraphDef. */
abstract class IMetaGraphDef {
  /** MetaGraphDef metaInfoDef */
  MetaGraphDef_IMetaInfoDef? metaInfoDef;

  /** MetaGraphDef graphDef */
  IGraphDef? graphDef;

  /** MetaGraphDef saverDef */
  ISaverDef? saverDef;

  /** MetaGraphDef collectionDef */
  Map<String, ICollectionDef>? collectionDef;

  /** MetaGraphDef signatureDef */
  Map<String, ISignatureDef>? signatureDef;

  /** MetaGraphDef assetFileDef */
  List<IAssetFileDef>? assetFileDef;
}

// class MetaGraphDef {
/** Properties of a MetaInfoDef. */
class MetaGraphDef_IMetaInfoDef {
  /** MetaInfoDef metaGraphVersion */
  String? metaGraphVersion;

  /** MetaInfoDef strippedOpList */
  IOpList? strippedOpList;

  /** MetaInfoDef anyInfo */
  IAny? anyInfo;

  /** MetaInfoDef tags */
  List<String>? tags;

  /** MetaInfoDef tensorflowVersion */
  String? tensorflowVersion;

  /** MetaInfoDef tensorflowGitVersion */
  String? tensorflowGitVersion;
}
// }

/** Properties of a SavedModel. */
abstract class ISavedModel {
  /** SavedModel savedModelSchemaVersion */
  Object? savedModelSchemaVersion; // number|string|null

  /** SavedModel metaGraphs */
  List<IMetaGraphDef>? metaGraphs;
}

/** Properties of a FunctionDefLibrary. */
abstract class IFunctionDefLibrary {
  /** FunctionDefLibrary function */
  List<IFunctionDef>? function;

  /** FunctionDefLibrary gradient */
  List<IGradientDef>? gradient;
}

/** Properties of a FunctionDef. */
abstract class IFunctionDef {
  /** FunctionDef signature */
  IOpDef? signature;

  /** FunctionDef attr */
  Map<String, IAttrValue>? attr;

  /** FunctionDef nodeDef */
  List<INodeDef>? nodeDef;

  /** FunctionDef ret */
  Map<String, String>? ret;
}

/** Properties of a GradientDef. */
abstract class IGradientDef {
  /** GradientDef functionName */
  String? functionName;

  /** GradientDef gradientFunc */
  String? gradientFunc;
}
