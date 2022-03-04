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

import 'package:json_annotation/json_annotation.dart';

part 'compiled_api.g.dart';

/** Properties of an Any. */

@JsonSerializable()
class IAny {
  /** Any typeUrl */
  final String? typeUrl;

  /** Any value */
  @_UintListConvert
  final Uint8List? value;

  IAny({
    this.typeUrl,
    this.value,
  });

  Map<String, dynamic> toJson() => _$IAnyToJson(this);
  factory IAny.fromJson(Map<String, dynamic> map) => _$IAnyFromJson(map);
}

const _UintListConvert =
    JsonKey(fromJson: _uint8ListFromJson, toJson: _uint8ListToJson);
Uint8List? _uint8ListFromJson(Object? json) =>
    json is List ? Uint8List.fromList(json.cast()) : null;
List<int>? _uint8ListToJson(Uint8List? json) => json;

const _UintListListConvert =
    JsonKey(fromJson: _uint8ListListFromJson, toJson: _uint8ListListToJson);
List<Uint8List>? _uint8ListListFromJson(Object? json) => json is List
    ? json.map((e) => Uint8List.fromList(e.cast())).toList()
    : null;
List<List<int>>? _uint8ListListToJson(List<Uint8List>? json) => json;

/** DataType enum. */
@JsonEnum()
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  // @JsonValue(0)
  DT_INVALID, // = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  // @JsonValue(1)
  DT_FLOAT, // = 1,
  // @JsonValue(2)
  DT_DOUBLE, // = 2,
  // @JsonValue(3)
  DT_INT32, // = 3,
  // @JsonValue(4)
  DT_UINT8, // = 4,
  // @JsonValue(5)
  DT_INT16, // = 5,
  // @JsonValue(6)
  DT_INT8, // = 6,
  // @JsonValue(7)
  DT_STRING, // = 7,
  // @JsonValue(8)
  DT_COMPLEX64, // = 8,  // Single-precision complex
  // @JsonValue(9)
  DT_INT64, // = 9,
  // @JsonValue(10)
  DT_BOOL, // = 10,
  // @JsonValue(11)
  DT_QINT8, // = 11,     // Quantized int8
  // @JsonValue(12)
  DT_QUINT8, // = 12,    // Quantized uint8
  // @JsonValue(13)
  DT_QINT32, // = 13,    // Quantized int32
  // @JsonValue(14)
  DT_BFLOAT16, // = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  // @JsonValue(15)
  DT_QINT16, // = 15,    // Quantized int16
  // @JsonValue(16)
  DT_QUINT16, // = 16,   // Quantized uint16
  // @JsonValue(17)
  DT_UINT16, // = 17,
  // @JsonValue(18)
  DT_COMPLEX128, // = 18,  // Double-precision complex
  // @JsonValue(19)
  DT_HALF, // = 19,
  // @JsonValue(20)
  DT_RESOURCE, // = 20,
  // @JsonValue(21)
  DT_VARIANT, // = 21,  // Arbitrary C++ data types
  // @JsonValue(22)
  DT_UINT32, // = 22,
  // @JsonValue(23)
  DT_UINT64, // = 23,

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  // @JsonValue(101)
  DT_FLOAT_REF, // = 101,
  // @JsonValue(102)
  DT_DOUBLE_REF, // = 102,
  // @JsonValue(103)
  DT_INT32_REF, // = 103,
  // @JsonValue(104)
  DT_UINT8_REF, // = 104,
  // @JsonValue(105)
  DT_INT16_REF, // = 105,
  // @JsonValue(106)
  DT_INT8_REF, // = 106,
  // @JsonValue(107)
  DT_STRING_REF, // = 107,
  // @JsonValue(108)
  DT_COMPLEX64_REF, // = 108,
  // @JsonValue(109)
  DT_INT64_REF, // = 109,
  // @JsonValue(110)
  DT_BOOL_REF, // = 110,
  // @JsonValue(111)
  DT_QINT8_REF, // = 111,
  // @JsonValue(112)
  DT_QUINT8_REF, // = 112,
  // @JsonValue(113)
  DT_QINT32_REF, // = 113,
  // @JsonValue(114)
  DT_BFLOAT16_REF, // = 114,
  // @JsonValue(115)
  DT_QINT16_REF, // = 115,
  // @JsonValue(116)
  DT_QUINT16_REF, // = 116,
  // @JsonValue(117)
  DT_UINT16_REF, // = 117,
  // @JsonValue(118)
  DT_COMPLEX128_REF, // = 118,
  // @JsonValue(119)
  DT_HALF_REF, // = 119,
  // @JsonValue(120)
  DT_RESOURCE_REF, // = 120,
  // @JsonValue(121)
  DT_VARIANT_REF, // = 121,
  // @JsonValue(122)
  DT_UINT32_REF, // = 122,
  // @JsonValue(123)
  DT_UINT64_REF, // = 123,
}

/** Properties of a TensorShape. */
@JsonSerializable()
class ITensorShape {
  /** TensorShape dim */
  final List<TensorShape_IDim>? dim;

  /** TensorShape unknownRank */
  final bool? unknownRank;

  ITensorShape({
    this.dim,
    this.unknownRank,
  });

  Map<String, dynamic> toJson() => _$ITensorShapeToJson(this);
  factory ITensorShape.fromJson(Map<String, dynamic> map) =>
      _$ITensorShapeFromJson(map);
}

// class TensorShape {
/** Properties of a Dim. */
@JsonSerializable()
class TensorShape_IDim {
  /** Dim size */
  final Object? size; // number|string

  /** Dim name */
  final String? name;

  TensorShape_IDim({
    this.size,
    this.name,
  });

  Map<String, dynamic> toJson() => _$TensorShape_IDimToJson(this);
  factory TensorShape_IDim.fromJson(Map<String, dynamic> map) =>
      _$TensorShape_IDimFromJson(map);
}
// }

/** Properties of a Tensor. */
@JsonSerializable()
class ITensor {
  /** Tensor dtype */
  final DataType? dtype;

  /** Tensor tensorShape */
  final ITensorShape? tensorShape;

  /** Tensor versionNumber */
  final int? versionNumber;

  /** Tensor tensorContent */
  @_UintListConvert
  final Uint8List? tensorContent;

  /** Tensor floatVal */
  final List<int>? floatVal;

  /** Tensor doubleVal */
  final List<int>? doubleVal;

  /** Tensor intVal */
  final List<int>? intVal;

  /** Tensor stringVal */
  @_UintListListConvert
  final List<Uint8List>? stringVal;

  /** Tensor scomplexVal */
  final List<int>? scomplexVal;

  /** Tensor int64Val */
  final List<Object?>? int64Val; // (number | string)[]

  /** Tensor boolVal */
  final List<bool>? boolVal;

  /** Tensor uint32Val */
  final List<int>? uint32Val;

  /** Tensor uint64Val */
  final List<Object?>? uint64Val; // (number | string)[]

  ITensor({
    this.dtype,
    this.tensorShape,
    this.versionNumber,
    this.tensorContent,
    this.floatVal,
    this.doubleVal,
    this.intVal,
    this.stringVal,
    this.scomplexVal,
    this.int64Val,
    this.boolVal,
    this.uint32Val,
    this.uint64Val,
  });

  Map<String, dynamic> toJson() => _$ITensorToJson(this);
  factory ITensor.fromJson(Map<String, dynamic> map) => _$ITensorFromJson(map);
}

/** Properties of an AttrValue. */
@JsonSerializable()
class IAttrValue {
  /** AttrValue list */
  final AttrValue_IListValue? list;

  /** AttrValue s */
  final String? s;

  /** AttrValue i */
  final Object? i; // number|string

  /** AttrValue f */
  final double? f;

  /** AttrValue b */
  final bool? b;

  /** AttrValue type */
  final DataType? type;

  /** AttrValue shape */
  final ITensorShape? shape;

  /** AttrValue tensor */
  final ITensor? tensor;

  /** AttrValue placeholder */
  final String? placeholder;

  /** AttrValue func */
  final INameAttrList? func;

  IAttrValue({
    this.list,
    this.s,
    this.i,
    this.f,
    this.b,
    this.type,
    this.shape,
    this.tensor,
    this.placeholder,
    this.func,
  });

  Map<String, dynamic> toJson() => _$IAttrValueToJson(this);
  factory IAttrValue.fromJson(Map<String, dynamic> map) =>
      _$IAttrValueFromJson(map);
}

// class AttrValue {
/** Properties of a ListValue. */
@JsonSerializable()
class AttrValue_IListValue {
  /** ListValue s */
  final List<String>? s;

  /** ListValue i */
  final List<Object>? i; // (number | string)

  /** ListValue f */
  final List<int>? f;

  /** ListValue b */
  final List<bool>? b;

  /** ListValue type */
  final List<DataType>? type;

  /** ListValue shape */
  final List<ITensorShape>? shape;

  /** ListValue tensor */
  final List<ITensor>? tensor;

  /** ListValue func */
  final List<INameAttrList>? func;

  AttrValue_IListValue({
    this.s,
    this.i,
    this.f,
    this.b,
    this.type,
    this.shape,
    this.tensor,
    this.func,
  });

  Map<String, dynamic> toJson() => _$AttrValue_IListValueToJson(this);
  factory AttrValue_IListValue.fromJson(Map<String, dynamic> map) =>
      _$AttrValue_IListValueFromJson(map);
}

// }

/** Properties of a NameAttrList. */
class INameAttrList {
  /** NameAttrList name */
  final String? name;

  /** NameAttrList attr */
  final Map<String, IAttrValue>? attr;

  INameAttrList({
    this.name,
    this.attr,
  });

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'attr': attr,
    };
  }

  factory INameAttrList.fromJson(Map<String, dynamic> map) {
    return INameAttrList(
      name: map['name'],
      attr: (map['attr'] as Map?)
          ?.map((key, value) => MapEntry(key, IAttrValue.fromJson(value))),
    );
  }
}

/** Properties of a NodeDef. */
@JsonSerializable()
class INodeDef {
  /** NodeDef name */
  final String? name;

  /** NodeDef op */
  final String? op;

  /** NodeDef input */
  final List<String>? input;

  /** NodeDef device */
  final String? device;

  /** NodeDef attr */
  final Map<String, IAttrValue>? attr;

  INodeDef({
    this.name,
    this.op,
    this.input,
    this.device,
    this.attr,
  });

  Map<String, dynamic> toJson() => _$INodeDefToJson(this);
  factory INodeDef.fromJson(Map<String, dynamic> map) =>
      _$INodeDefFromJson(map);
}

/** Properties of a VersionDef. */
class IVersionDef {
  /** VersionDef producer */
  final int? producer;

  /** VersionDef minConsumer */
  final int? minConsumer;

  /** VersionDef badConsumers */
  final List<int>? badConsumers;
  IVersionDef({
    this.producer,
    this.minConsumer,
    this.badConsumers,
  });

  Map<String, dynamic> toJson() {
    return {
      'producer': producer,
      'minConsumer': minConsumer,
      'badConsumers': badConsumers,
    };
  }

  factory IVersionDef.fromJson(Map<String, dynamic> map) {
    return IVersionDef(
      producer: map['producer']?.toInt(),
      minConsumer: map['minConsumer']?.toInt(),
      badConsumers: map['badConsumers'] == null
          ? null
          : List<int>.from(map['badConsumers']),
    );
  }
}

/** Properties of a GraphDef. */
class IGraphDef {
  /** GraphDef node */
  final List<INodeDef>? node;

  /** GraphDef versions */
  final IVersionDef? versions;

  /** GraphDef library */
  final IFunctionDefLibrary? library;

  IGraphDef({
    this.node,
    this.versions,
    this.library,
  });

  Map<String, dynamic> toJson() {
    return {
      'node': node?.map((x) => x.toJson()).toList(),
      'versions': versions?.toJson(),
      'library': library?.toJson(),
    };
  }

  factory IGraphDef.fromJson(Map<String, dynamic> map) {
    return IGraphDef(
      node: map['node'] != null
          ? List<INodeDef>.from(map['node']?.map((x) => INodeDef.fromJson(x)))
          : null,
      versions: map['versions'] != null
          ? IVersionDef.fromJson(map['versions'])
          : null,
      library: map['library'] != null
          ? IFunctionDefLibrary.fromJson(map['library'])
          : null,
    );
  }
}

/** Properties of a CollectionDef. */
@JsonSerializable()
class ICollectionDef {
  /** CollectionDef nodeList */
  final CollectionDef_INodeList? nodeList;

  /** CollectionDef bytesList */
  final CollectionDef_IBytesList? bytesList;

  /** CollectionDef int64List */
  final CollectionDef_IInt64List? int64List;

  /** CollectionDef floatList */
  final CollectionDef_IFloatList? floatList;

  /** CollectionDef anyList */
  final CollectionDef_IAnyList? anyList;

  ICollectionDef({
    this.nodeList,
    this.bytesList,
    this.int64List,
    this.floatList,
    this.anyList,
  });

  Map<String, dynamic> toJson() => _$ICollectionDefToJson(this);
  factory ICollectionDef.fromJson(Map<String, dynamic> map) =>
      _$ICollectionDefFromJson(map);
}

// class CollectionDef {
/** Properties of a NodeList. */
@JsonSerializable()
class CollectionDef_INodeList {
  /** NodeList value */
  final List<String>? value;

  CollectionDef_INodeList({
    this.value,
  });

  Map<String, dynamic> toJson() => _$CollectionDef_INodeListToJson(this);
  factory CollectionDef_INodeList.fromJson(Map<String, dynamic> map) =>
      _$CollectionDef_INodeListFromJson(map);
}

/** Properties of a BytesList. */
class CollectionDef_IBytesList {
  /** BytesList value */
  final List<Uint8List>? value;

  CollectionDef_IBytesList({
    this.value,
  });

  Map<String, Object?> toJson() => {'value': value};
  factory CollectionDef_IBytesList.fromJson(Map<String, Object?> json) =>
      CollectionDef_IBytesList(
        value: json['value'] != null
            ? List<Uint8List>.from(
                (json['value'] as List).map((x) => Uint8List.fromList(x)))
            : null,
      );
}

/** Properties of an Int64List. */
@JsonSerializable()
class CollectionDef_IInt64List {
  /** Int64List value */
  final List<Object>? value; // number | string
  CollectionDef_IInt64List({
    this.value,
  });

  Map<String, Object?> toJson() => _$CollectionDef_IInt64ListToJson(this);
  factory CollectionDef_IInt64List.fromJson(Map<String, Object?> json) =>
      _$CollectionDef_IInt64ListFromJson(json);
}

/** Properties of a FloatList. */
@JsonSerializable()
class CollectionDef_IFloatList {
  /** FloatList value */
  final List<int>? value;
  CollectionDef_IFloatList({
    this.value,
  });

  Map<String, Object?> toJson() => _$CollectionDef_IFloatListToJson(this);
  factory CollectionDef_IFloatList.fromJson(Map<String, Object?> json) =>
      _$CollectionDef_IFloatListFromJson(json);
}

/** Properties of an AnyList. */
@JsonSerializable()
class CollectionDef_IAnyList {
  /** AnyList value */
  final List<IAny>? value;
  CollectionDef_IAnyList({
    this.value,
  });

  Map<String, Object?> toJson() => _$CollectionDef_IAnyListToJson(this);
  factory CollectionDef_IAnyList.fromJson(Map<String, Object?> json) =>
      _$CollectionDef_IAnyListFromJson(json);
}
// }

/** Properties of a SaverDef. */
@JsonSerializable()
class ISaverDef {
  /** SaverDef filenameTensorName */
  final String? filenameTensorName;

  /** SaverDef saveTensorName */
  final String? saveTensorName;

  /** SaverDef restoreOpName */
  final String? restoreOpName;

  /** SaverDef maxToKeep */
  final int? maxToKeep;

  /** SaverDef sharded */
  final bool? sharded;

  /** SaverDef keepCheckpointEveryNHours */
  final int? keepCheckpointEveryNHours;

  /** SaverDef version */
  final SaverDef_CheckpointFormatVersion? version;

  ISaverDef({
    this.filenameTensorName,
    this.saveTensorName,
    this.restoreOpName,
    this.maxToKeep,
    this.sharded,
    this.keepCheckpointEveryNHours,
    this.version,
  });

  Map<String, dynamic> toJson() => _$ISaverDefToJson(this);
  factory ISaverDef.fromJson(Map<String, dynamic> map) =>
      _$ISaverDefFromJson(map);
}

// class SaverDef {
/** CheckpointFormatVersion enum. */
enum SaverDef_CheckpointFormatVersion { LEGACY, V1, V2 }
// }

/** Properties of a TensorInfo. */
@JsonSerializable()
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

  Map<String, dynamic> toJson() => _$ITensorInfoToJson(this);
  factory ITensorInfo.fromJson(Map<String, dynamic> map) =>
      _$ITensorInfoFromJson(map);
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

  Map<String, dynamic> toJson() {
    return {
      'valuesTensorName': valuesTensorName,
      'indicesTensorName': indicesTensorName,
      'denseShapeTensorName': denseShapeTensorName,
    };
  }

  factory TensorInfo_ICooSparse.fromJson(Map<String, dynamic> map) {
    return TensorInfo_ICooSparse(
      valuesTensorName: map['valuesTensorName'],
      indicesTensorName: map['indicesTensorName'],
      denseShapeTensorName: map['denseShapeTensorName'],
    );
  }
}
// }

/** Properties of a SignatureDef. */
@JsonSerializable()
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

  Map<String, dynamic> toJson() => _$ISignatureDefToJson(this);
  factory ISignatureDef.fromJson(Map<String, dynamic> map) =>
      _$ISignatureDefFromJson(map);
}

/** Properties of an AssetFileDef. */
@JsonSerializable()
class IAssetFileDef {
  /** AssetFileDef tensorInfo */
  final ITensorInfo? tensorInfo;

  /** AssetFileDef filename */
  final String? filename;

  IAssetFileDef({
    this.tensorInfo,
    this.filename,
  });

  Map<String, dynamic> toJson() => _$IAssetFileDefToJson(this);
  factory IAssetFileDef.fromJson(Map<String, dynamic> map) =>
      _$IAssetFileDefFromJson(map);
}

/** Properties of an OpDef. */
@JsonSerializable()
class IOpDef {
  /** OpDef name */
  final String? name;

  /** OpDef inputArg */
  final List<OpDef_IArgDef>? inputArg;

  /** OpDef outputArg */
  final List<OpDef_IArgDef>? outputArg;

  /** OpDef attr */
  final List<OpDef_IAttrDef>? attr;

  /** OpDef deprecation */
  final OpDef_IOpDeprecation? deprecation;

  /** OpDef summary */
  final String? summary;

  /** OpDef description */
  final String? description;

  /** OpDef isCommutative */
  final bool? isCommutative;

  /** OpDef isAggregate */
  final bool? isAggregate;

  /** OpDef isStateful */
  final bool? isStateful;

  /** OpDef allowsUninitializedInput */
  final bool? allowsUninitializedInput;

  IOpDef({
    this.name,
    this.inputArg,
    this.outputArg,
    this.attr,
    this.deprecation,
    this.summary,
    this.description,
    this.isCommutative,
    this.isAggregate,
    this.isStateful,
    this.allowsUninitializedInput,
  });

  Map<String, dynamic> toJson() => _$IOpDefToJson(this);
  factory IOpDef.fromJson(Map<String, dynamic> map) => _$IOpDefFromJson(map);
}

// class OpDef {
/** Properties of an ArgDef. */
@JsonSerializable()
class OpDef_IArgDef {
  /** ArgDef name */
  final String? name;

  /** ArgDef description */
  final String? description;

  /** ArgDef type */
  final DataType? type;

  /** ArgDef typeAttr */
  final String? typeAttr;

  /** ArgDef numberAttr */
  final String? numberAttr;

  /** ArgDef typeListAttr */
  final String? typeListAttr;

  /** ArgDef isRef */
  final bool? isRef;

  OpDef_IArgDef({
    this.name,
    this.description,
    this.type,
    this.typeAttr,
    this.numberAttr,
    this.typeListAttr,
    this.isRef,
  });

  Map<String, dynamic> toJson() => _$OpDef_IArgDefToJson(this);
  factory OpDef_IArgDef.fromJson(Map<String, dynamic> map) =>
      _$OpDef_IArgDefFromJson(map);
}

/** Properties of an AttrDef. */
@JsonSerializable()
class OpDef_IAttrDef {
  /** AttrDef name */
  final String? name;

  /** AttrDef type */
  final String? type;

  /** AttrDef defaultValue */
  final IAttrValue? defaultValue;

  /** AttrDef description */
  final String? description;

  /** AttrDef hasMinimum */
  final bool? hasMinimum;

  /** AttrDef minimum */
  final Object? minimum; // number|String

  /** AttrDef allowedValues */
  final IAttrValue? allowedValues;

  OpDef_IAttrDef({
    this.name,
    this.type,
    this.defaultValue,
    this.description,
    this.hasMinimum,
    this.minimum,
    this.allowedValues,
  });

  Map<String, dynamic> toJson() => _$OpDef_IAttrDefToJson(this);
  factory OpDef_IAttrDef.fromJson(Map<String, dynamic> map) =>
      _$OpDef_IAttrDefFromJson(map);
}

/** Properties of an OpDeprecation. */
@JsonSerializable()
class OpDef_IOpDeprecation {
  /** OpDeprecation version */
  final int? version;

  /** OpDeprecation explanation */
  final String? explanation;

  OpDef_IOpDeprecation({
    this.version,
    this.explanation,
  });

  Map<String, dynamic> toJson() => _$OpDef_IOpDeprecationToJson(this);
  factory OpDef_IOpDeprecation.fromJson(Map<String, dynamic> map) =>
      _$OpDef_IOpDeprecationFromJson(map);
}

// }

/** Properties of an OpList. */
@JsonSerializable()
class IOpList {
  /** OpList op */
  final List<IOpDef>? op;

  IOpList({
    this.op,
  });

  Map<String, dynamic> toJson() => _$IOpListToJson(this);
  factory IOpList.fromJson(Map<String, dynamic> map) => _$IOpListFromJson(map);
}

/** Properties of a MetaGraphDef. */
@JsonSerializable()
class IMetaGraphDef {
  /** MetaGraphDef metaInfoDef */
  final MetaGraphDef_IMetaInfoDef? metaInfoDef;

  /** MetaGraphDef graphDef */
  final IGraphDef? graphDef;

  /** MetaGraphDef saverDef */
  final ISaverDef? saverDef;

  /** MetaGraphDef collectionDef */
  final Map<String, ICollectionDef>? collectionDef;

  /** MetaGraphDef signatureDef */
  final Map<String, ISignatureDef>? signatureDef;

  /** MetaGraphDef assetFileDef */
  final List<IAssetFileDef>? assetFileDef;

  IMetaGraphDef({
    this.metaInfoDef,
    this.graphDef,
    this.saverDef,
    this.collectionDef,
    this.signatureDef,
    this.assetFileDef,
  });

  Map<String, dynamic> toJson() => _$IMetaGraphDefToJson(this);
  factory IMetaGraphDef.fromJson(Map<String, dynamic> map) =>
      _$IMetaGraphDefFromJson(map);
}

// class MetaGraphDef {
/** Properties of a MetaInfoDef. */
@JsonSerializable()
class MetaGraphDef_IMetaInfoDef {
  /** MetaInfoDef metaGraphVersion */
  final String? metaGraphVersion;

  /** MetaInfoDef strippedOpList */
  final IOpList? strippedOpList;

  /** MetaInfoDef anyInfo */
  final IAny? anyInfo;

  /** MetaInfoDef tags */
  final List<String>? tags;

  /** MetaInfoDef tensorflowVersion */
  final String? tensorflowVersion;

  /** MetaInfoDef tensorflowGitVersion */
  final String? tensorflowGitVersion;

  MetaGraphDef_IMetaInfoDef({
    this.metaGraphVersion,
    this.strippedOpList,
    this.anyInfo,
    this.tags,
    this.tensorflowVersion,
    this.tensorflowGitVersion,
  });

  Map<String, dynamic> toJson() => _$MetaGraphDef_IMetaInfoDefToJson(this);
  factory MetaGraphDef_IMetaInfoDef.fromJson(Map<String, dynamic> map) =>
      _$MetaGraphDef_IMetaInfoDefFromJson(map);
}
// }

/** Properties of a SavedModel. */
@JsonSerializable()
class ISavedModel {
  /** SavedModel savedModelSchemaVersion */
  final Object? savedModelSchemaVersion; // number|string|null

  /** SavedModel metaGraphs */
  final List<IMetaGraphDef>? metaGraphs;

  ISavedModel({
    this.savedModelSchemaVersion,
    this.metaGraphs,
  });

  Map<String, dynamic> toJson() => _$ISavedModelToJson(this);
  factory ISavedModel.fromJson(Map<String, dynamic> map) =>
      _$ISavedModelFromJson(map);
}

/** Properties of a FunctionDefLibrary. */
@JsonSerializable()
class IFunctionDefLibrary {
  /** FunctionDefLibrary function */
  final List<IFunctionDef>? function;

  /** FunctionDefLibrary gradient */
  final List<IGradientDef>? gradient;

  IFunctionDefLibrary({
    this.function,
    this.gradient,
  });

  Map<String, dynamic> toJson() => _$IFunctionDefLibraryToJson(this);
  factory IFunctionDefLibrary.fromJson(Map<String, dynamic> map) =>
      _$IFunctionDefLibraryFromJson(map);
}

/** Properties of a FunctionDef. */
@JsonSerializable()
class IFunctionDef {
  /** FunctionDef signature */
  final IOpDef? signature;

  /** FunctionDef attr */
  final Map<String, IAttrValue>? attr;

  /** FunctionDef nodeDef */
  final List<INodeDef>? nodeDef;

  /** FunctionDef ret */
  final Map<String, String>? ret;

  IFunctionDef({
    this.signature,
    this.attr,
    this.nodeDef,
    this.ret,
  });

  Map<String, dynamic> toJson() => _$IFunctionDefToJson(this);
  factory IFunctionDef.fromJson(Map<String, dynamic> map) =>
      _$IFunctionDefFromJson(map);
}

/** Properties of a GradientDef. */
@JsonSerializable()
class IGradientDef {
  /** GradientDef functionName */
  final String? functionName;

  /** GradientDef gradientFunc */
  final String? gradientFunc;

  IGradientDef({
    this.functionName,
    this.gradientFunc,
  });

  Map<String, dynamic> toJson() => _$IGradientDefToJson(this);
  factory IGradientDef.fromJson(Map<String, dynamic> map) =>
      _$IGradientDefFromJson(map);
}
