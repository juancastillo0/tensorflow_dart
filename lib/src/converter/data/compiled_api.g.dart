// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'compiled_api.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

IAny _$IAnyFromJson(Map<String, dynamic> json) => IAny(
      typeUrl: json['typeUrl'] as String?,
      value: _uint8ListFromJson(json['value']),
    );

Map<String, dynamic> _$IAnyToJson(IAny instance) => <String, dynamic>{
      'typeUrl': instance.typeUrl,
      'value': _uint8ListToJson(instance.value),
    };

ITensorShape _$ITensorShapeFromJson(Map<String, dynamic> json) => ITensorShape(
      dim: (json['dim'] as List<dynamic>?)
          ?.map((e) => TensorShape_IDim.fromJson(e as Map<String, dynamic>))
          .toList(),
      unknownRank: json['unknownRank'] as bool?,
    );

Map<String, dynamic> _$ITensorShapeToJson(ITensorShape instance) =>
    <String, dynamic>{
      'dim': instance.dim,
      'unknownRank': instance.unknownRank,
    };

TensorShape_IDim _$TensorShape_IDimFromJson(Map<String, dynamic> json) =>
    TensorShape_IDim(
      size: json['size'],
      name: json['name'] as String?,
    );

Map<String, dynamic> _$TensorShape_IDimToJson(TensorShape_IDim instance) =>
    <String, dynamic>{
      'size': instance.size,
      'name': instance.name,
    };

ITensor _$ITensorFromJson(Map<String, dynamic> json) => ITensor(
      dtype: $enumDecodeNullable(_$DataTypeEnumMap, json['dtype']),
      tensorShape: json['tensorShape'] == null
          ? null
          : ITensorShape.fromJson(json['tensorShape'] as Map<String, dynamic>),
      versionNumber: json['versionNumber'] as int?,
      tensorContent: _uint8ListFromJson(json['tensorContent']),
      floatVal:
          (json['floatVal'] as List<dynamic>?)?.map((e) => e as int).toList(),
      doubleVal:
          (json['doubleVal'] as List<dynamic>?)?.map((e) => e as int).toList(),
      intVal: (json['intVal'] as List<dynamic>?)?.map((e) => e as int).toList(),
      stringVal: _uint8ListListFromJson(json['stringVal']),
      scomplexVal: (json['scomplexVal'] as List<dynamic>?)
          ?.map((e) => e as int)
          .toList(),
      int64Val: json['int64Val'] as List<dynamic>?,
      boolVal:
          (json['boolVal'] as List<dynamic>?)?.map((e) => e as bool).toList(),
      uint32Val:
          (json['uint32Val'] as List<dynamic>?)?.map((e) => e as int).toList(),
      uint64Val: json['uint64Val'] as List<dynamic>?,
    );

Map<String, dynamic> _$ITensorToJson(ITensor instance) => <String, dynamic>{
      'dtype': _$DataTypeEnumMap[instance.dtype],
      'tensorShape': instance.tensorShape,
      'versionNumber': instance.versionNumber,
      'tensorContent': _uint8ListToJson(instance.tensorContent),
      'floatVal': instance.floatVal,
      'doubleVal': instance.doubleVal,
      'intVal': instance.intVal,
      'stringVal': _uint8ListListToJson(instance.stringVal),
      'scomplexVal': instance.scomplexVal,
      'int64Val': instance.int64Val,
      'boolVal': instance.boolVal,
      'uint32Val': instance.uint32Val,
      'uint64Val': instance.uint64Val,
    };

const _$DataTypeEnumMap = {
  DataType.DT_INVALID: 'DT_INVALID',
  DataType.DT_FLOAT: 'DT_FLOAT',
  DataType.DT_DOUBLE: 'DT_DOUBLE',
  DataType.DT_INT32: 'DT_INT32',
  DataType.DT_UINT8: 'DT_UINT8',
  DataType.DT_INT16: 'DT_INT16',
  DataType.DT_INT8: 'DT_INT8',
  DataType.DT_STRING: 'DT_STRING',
  DataType.DT_COMPLEX64: 'DT_COMPLEX64',
  DataType.DT_INT64: 'DT_INT64',
  DataType.DT_BOOL: 'DT_BOOL',
  DataType.DT_QINT8: 'DT_QINT8',
  DataType.DT_QUINT8: 'DT_QUINT8',
  DataType.DT_QINT32: 'DT_QINT32',
  DataType.DT_BFLOAT16: 'DT_BFLOAT16',
  DataType.DT_QINT16: 'DT_QINT16',
  DataType.DT_QUINT16: 'DT_QUINT16',
  DataType.DT_UINT16: 'DT_UINT16',
  DataType.DT_COMPLEX128: 'DT_COMPLEX128',
  DataType.DT_HALF: 'DT_HALF',
  DataType.DT_RESOURCE: 'DT_RESOURCE',
  DataType.DT_VARIANT: 'DT_VARIANT',
  DataType.DT_UINT32: 'DT_UINT32',
  DataType.DT_UINT64: 'DT_UINT64',
  DataType.DT_FLOAT_REF: 'DT_FLOAT_REF',
  DataType.DT_DOUBLE_REF: 'DT_DOUBLE_REF',
  DataType.DT_INT32_REF: 'DT_INT32_REF',
  DataType.DT_UINT8_REF: 'DT_UINT8_REF',
  DataType.DT_INT16_REF: 'DT_INT16_REF',
  DataType.DT_INT8_REF: 'DT_INT8_REF',
  DataType.DT_STRING_REF: 'DT_STRING_REF',
  DataType.DT_COMPLEX64_REF: 'DT_COMPLEX64_REF',
  DataType.DT_INT64_REF: 'DT_INT64_REF',
  DataType.DT_BOOL_REF: 'DT_BOOL_REF',
  DataType.DT_QINT8_REF: 'DT_QINT8_REF',
  DataType.DT_QUINT8_REF: 'DT_QUINT8_REF',
  DataType.DT_QINT32_REF: 'DT_QINT32_REF',
  DataType.DT_BFLOAT16_REF: 'DT_BFLOAT16_REF',
  DataType.DT_QINT16_REF: 'DT_QINT16_REF',
  DataType.DT_QUINT16_REF: 'DT_QUINT16_REF',
  DataType.DT_UINT16_REF: 'DT_UINT16_REF',
  DataType.DT_COMPLEX128_REF: 'DT_COMPLEX128_REF',
  DataType.DT_HALF_REF: 'DT_HALF_REF',
  DataType.DT_RESOURCE_REF: 'DT_RESOURCE_REF',
  DataType.DT_VARIANT_REF: 'DT_VARIANT_REF',
  DataType.DT_UINT32_REF: 'DT_UINT32_REF',
  DataType.DT_UINT64_REF: 'DT_UINT64_REF',
};

IAttrValue _$IAttrValueFromJson(Map<String, dynamic> json) => IAttrValue(
      list: json['list'] == null
          ? null
          : AttrValue_IListValue.fromJson(json['list'] as Map<String, dynamic>),
      s: json['s'] as String?,
      i: json['i'],
      f: (json['f'] as num?)?.toDouble(),
      b: json['b'] as bool?,
      type: $enumDecodeNullable(_$DataTypeEnumMap, json['type']),
      shape: json['shape'] == null
          ? null
          : ITensorShape.fromJson(json['shape'] as Map<String, dynamic>),
      tensor: json['tensor'] == null
          ? null
          : ITensor.fromJson(json['tensor'] as Map<String, dynamic>),
      placeholder: json['placeholder'] as String?,
      func: json['func'] == null
          ? null
          : INameAttrList.fromJson(json['func'] as Map<String, dynamic>),
    );

Map<String, dynamic> _$IAttrValueToJson(IAttrValue instance) =>
    <String, dynamic>{
      'list': instance.list,
      's': instance.s,
      'i': instance.i,
      'f': instance.f,
      'b': instance.b,
      'type': _$DataTypeEnumMap[instance.type],
      'shape': instance.shape,
      'tensor': instance.tensor,
      'placeholder': instance.placeholder,
      'func': instance.func,
    };

AttrValue_IListValue _$AttrValue_IListValueFromJson(
        Map<String, dynamic> json) =>
    AttrValue_IListValue(
      s: (json['s'] as List<dynamic>?)?.map((e) => e as String).toList(),
      i: (json['i'] as List<dynamic>?)?.map((e) => e as Object).toList(),
      f: (json['f'] as List<dynamic>?)?.map((e) => e as int).toList(),
      b: (json['b'] as List<dynamic>?)?.map((e) => e as bool).toList(),
      type: (json['type'] as List<dynamic>?)
          ?.map((e) => $enumDecode(_$DataTypeEnumMap, e))
          .toList(),
      shape: (json['shape'] as List<dynamic>?)
          ?.map((e) => ITensorShape.fromJson(e as Map<String, dynamic>))
          .toList(),
      tensor: (json['tensor'] as List<dynamic>?)
          ?.map((e) => ITensor.fromJson(e as Map<String, dynamic>))
          .toList(),
      func: (json['func'] as List<dynamic>?)
          ?.map((e) => INameAttrList.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$AttrValue_IListValueToJson(
        AttrValue_IListValue instance) =>
    <String, dynamic>{
      's': instance.s,
      'i': instance.i,
      'f': instance.f,
      'b': instance.b,
      'type': instance.type?.map((e) => _$DataTypeEnumMap[e]).toList(),
      'shape': instance.shape,
      'tensor': instance.tensor,
      'func': instance.func,
    };

INodeDef _$INodeDefFromJson(Map<String, dynamic> json) => INodeDef(
      name: json['name'] as String?,
      op: json['op'] as String?,
      input:
          (json['input'] as List<dynamic>?)?.map((e) => e as String).toList(),
      device: json['device'] as String?,
      attr: (json['attr'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, IAttrValue.fromJson(e as Map<String, dynamic>)),
      ),
    );

Map<String, dynamic> _$INodeDefToJson(INodeDef instance) => <String, dynamic>{
      'name': instance.name,
      'op': instance.op,
      'input': instance.input,
      'device': instance.device,
      'attr': instance.attr,
    };

ICollectionDef _$ICollectionDefFromJson(Map<String, dynamic> json) =>
    ICollectionDef(
      nodeList: json['nodeList'] == null
          ? null
          : CollectionDef_INodeList.fromJson(
              json['nodeList'] as Map<String, dynamic>),
      bytesList: json['bytesList'] == null
          ? null
          : CollectionDef_IBytesList.fromJson(
              json['bytesList'] as Map<String, dynamic>),
      int64List: json['int64List'] == null
          ? null
          : CollectionDef_IInt64List.fromJson(
              json['int64List'] as Map<String, dynamic>),
      floatList: json['floatList'] == null
          ? null
          : CollectionDef_IFloatList.fromJson(
              json['floatList'] as Map<String, dynamic>),
      anyList: json['anyList'] == null
          ? null
          : CollectionDef_IAnyList.fromJson(
              json['anyList'] as Map<String, dynamic>),
    );

Map<String, dynamic> _$ICollectionDefToJson(ICollectionDef instance) =>
    <String, dynamic>{
      'nodeList': instance.nodeList,
      'bytesList': instance.bytesList,
      'int64List': instance.int64List,
      'floatList': instance.floatList,
      'anyList': instance.anyList,
    };

CollectionDef_INodeList _$CollectionDef_INodeListFromJson(
        Map<String, dynamic> json) =>
    CollectionDef_INodeList(
      value:
          (json['value'] as List<dynamic>?)?.map((e) => e as String).toList(),
    );

Map<String, dynamic> _$CollectionDef_INodeListToJson(
        CollectionDef_INodeList instance) =>
    <String, dynamic>{
      'value': instance.value,
    };

CollectionDef_IInt64List _$CollectionDef_IInt64ListFromJson(
        Map<String, dynamic> json) =>
    CollectionDef_IInt64List(
      value:
          (json['value'] as List<dynamic>?)?.map((e) => e as Object).toList(),
    );

Map<String, dynamic> _$CollectionDef_IInt64ListToJson(
        CollectionDef_IInt64List instance) =>
    <String, dynamic>{
      'value': instance.value,
    };

CollectionDef_IFloatList _$CollectionDef_IFloatListFromJson(
        Map<String, dynamic> json) =>
    CollectionDef_IFloatList(
      value: (json['value'] as List<dynamic>?)?.map((e) => e as int).toList(),
    );

Map<String, dynamic> _$CollectionDef_IFloatListToJson(
        CollectionDef_IFloatList instance) =>
    <String, dynamic>{
      'value': instance.value,
    };

CollectionDef_IAnyList _$CollectionDef_IAnyListFromJson(
        Map<String, dynamic> json) =>
    CollectionDef_IAnyList(
      value: (json['value'] as List<dynamic>?)
          ?.map((e) => IAny.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$CollectionDef_IAnyListToJson(
        CollectionDef_IAnyList instance) =>
    <String, dynamic>{
      'value': instance.value,
    };

ISaverDef _$ISaverDefFromJson(Map<String, dynamic> json) => ISaverDef(
      filenameTensorName: json['filenameTensorName'] as String?,
      saveTensorName: json['saveTensorName'] as String?,
      restoreOpName: json['restoreOpName'] as String?,
      maxToKeep: json['maxToKeep'] as int?,
      sharded: json['sharded'] as bool?,
      keepCheckpointEveryNHours: json['keepCheckpointEveryNHours'] as int?,
      version: $enumDecodeNullable(
          _$SaverDef_CheckpointFormatVersionEnumMap, json['version']),
    );

Map<String, dynamic> _$ISaverDefToJson(ISaverDef instance) => <String, dynamic>{
      'filenameTensorName': instance.filenameTensorName,
      'saveTensorName': instance.saveTensorName,
      'restoreOpName': instance.restoreOpName,
      'maxToKeep': instance.maxToKeep,
      'sharded': instance.sharded,
      'keepCheckpointEveryNHours': instance.keepCheckpointEveryNHours,
      'version': _$SaverDef_CheckpointFormatVersionEnumMap[instance.version],
    };

const _$SaverDef_CheckpointFormatVersionEnumMap = {
  SaverDef_CheckpointFormatVersion.LEGACY: 'LEGACY',
  SaverDef_CheckpointFormatVersion.V1: 'V1',
  SaverDef_CheckpointFormatVersion.V2: 'V2',
};

ITensorInfo _$ITensorInfoFromJson(Map<String, dynamic> json) => ITensorInfo(
      name: json['name'] as String?,
      cooSparse: json['cooSparse'] == null
          ? null
          : TensorInfo_ICooSparse.fromJson(
              json['cooSparse'] as Map<String, dynamic>),
      dtype: $enumDecodeNullable(_$DataTypeEnumMap, json['dtype']),
      tensorShape: json['tensorShape'] == null
          ? null
          : ITensorShape.fromJson(json['tensorShape'] as Map<String, dynamic>),
    );

Map<String, dynamic> _$ITensorInfoToJson(ITensorInfo instance) =>
    <String, dynamic>{
      'name': instance.name,
      'cooSparse': instance.cooSparse,
      'dtype': _$DataTypeEnumMap[instance.dtype],
      'tensorShape': instance.tensorShape,
    };

ISignatureDef _$ISignatureDefFromJson(Map<String, dynamic> json) =>
    ISignatureDef(
      inputs: (json['inputs'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, ITensorInfo.fromJson(e as Map<String, dynamic>)),
      ),
      outputs: (json['outputs'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, ITensorInfo.fromJson(e as Map<String, dynamic>)),
      ),
      methodName: json['methodName'] as String?,
    );

Map<String, dynamic> _$ISignatureDefToJson(ISignatureDef instance) =>
    <String, dynamic>{
      'inputs': instance.inputs,
      'outputs': instance.outputs,
      'methodName': instance.methodName,
    };

IAssetFileDef _$IAssetFileDefFromJson(Map<String, dynamic> json) =>
    IAssetFileDef(
      tensorInfo: json['tensorInfo'] == null
          ? null
          : ITensorInfo.fromJson(json['tensorInfo'] as Map<String, dynamic>),
      filename: json['filename'] as String?,
    );

Map<String, dynamic> _$IAssetFileDefToJson(IAssetFileDef instance) =>
    <String, dynamic>{
      'tensorInfo': instance.tensorInfo,
      'filename': instance.filename,
    };

IOpDef _$IOpDefFromJson(Map<String, dynamic> json) => IOpDef(
      name: json['name'] as String?,
      inputArg: (json['inputArg'] as List<dynamic>?)
          ?.map((e) => OpDef_IArgDef.fromJson(e as Map<String, dynamic>))
          .toList(),
      outputArg: (json['outputArg'] as List<dynamic>?)
          ?.map((e) => OpDef_IArgDef.fromJson(e as Map<String, dynamic>))
          .toList(),
      attr: (json['attr'] as List<dynamic>?)
          ?.map((e) => OpDef_IAttrDef.fromJson(e as Map<String, dynamic>))
          .toList(),
      deprecation: json['deprecation'] == null
          ? null
          : OpDef_IOpDeprecation.fromJson(
              json['deprecation'] as Map<String, dynamic>),
      summary: json['summary'] as String?,
      description: json['description'] as String?,
      isCommutative: json['isCommutative'] as bool?,
      isAggregate: json['isAggregate'] as bool?,
      isStateful: json['isStateful'] as bool?,
      allowsUninitializedInput: json['allowsUninitializedInput'] as bool?,
    );

Map<String, dynamic> _$IOpDefToJson(IOpDef instance) => <String, dynamic>{
      'name': instance.name,
      'inputArg': instance.inputArg,
      'outputArg': instance.outputArg,
      'attr': instance.attr,
      'deprecation': instance.deprecation,
      'summary': instance.summary,
      'description': instance.description,
      'isCommutative': instance.isCommutative,
      'isAggregate': instance.isAggregate,
      'isStateful': instance.isStateful,
      'allowsUninitializedInput': instance.allowsUninitializedInput,
    };

OpDef_IArgDef _$OpDef_IArgDefFromJson(Map<String, dynamic> json) =>
    OpDef_IArgDef(
      name: json['name'] as String?,
      description: json['description'] as String?,
      type: $enumDecodeNullable(_$DataTypeEnumMap, json['type']),
      typeAttr: json['typeAttr'] as String?,
      numberAttr: json['numberAttr'] as String?,
      typeListAttr: json['typeListAttr'] as String?,
      isRef: json['isRef'] as bool?,
    );

Map<String, dynamic> _$OpDef_IArgDefToJson(OpDef_IArgDef instance) =>
    <String, dynamic>{
      'name': instance.name,
      'description': instance.description,
      'type': _$DataTypeEnumMap[instance.type],
      'typeAttr': instance.typeAttr,
      'numberAttr': instance.numberAttr,
      'typeListAttr': instance.typeListAttr,
      'isRef': instance.isRef,
    };

OpDef_IAttrDef _$OpDef_IAttrDefFromJson(Map<String, dynamic> json) =>
    OpDef_IAttrDef(
      name: json['name'] as String?,
      type: json['type'] as String?,
      defaultValue: json['defaultValue'] == null
          ? null
          : IAttrValue.fromJson(json['defaultValue'] as Map<String, dynamic>),
      description: json['description'] as String?,
      hasMinimum: json['hasMinimum'] as bool?,
      minimum: json['minimum'],
      allowedValues: json['allowedValues'] == null
          ? null
          : IAttrValue.fromJson(json['allowedValues'] as Map<String, dynamic>),
    );

Map<String, dynamic> _$OpDef_IAttrDefToJson(OpDef_IAttrDef instance) =>
    <String, dynamic>{
      'name': instance.name,
      'type': instance.type,
      'defaultValue': instance.defaultValue,
      'description': instance.description,
      'hasMinimum': instance.hasMinimum,
      'minimum': instance.minimum,
      'allowedValues': instance.allowedValues,
    };

OpDef_IOpDeprecation _$OpDef_IOpDeprecationFromJson(
        Map<String, dynamic> json) =>
    OpDef_IOpDeprecation(
      version: json['version'] as int?,
      explanation: json['explanation'] as String?,
    );

Map<String, dynamic> _$OpDef_IOpDeprecationToJson(
        OpDef_IOpDeprecation instance) =>
    <String, dynamic>{
      'version': instance.version,
      'explanation': instance.explanation,
    };

IOpList _$IOpListFromJson(Map<String, dynamic> json) => IOpList(
      op: (json['op'] as List<dynamic>?)
          ?.map((e) => IOpDef.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$IOpListToJson(IOpList instance) => <String, dynamic>{
      'op': instance.op,
    };

IMetaGraphDef _$IMetaGraphDefFromJson(Map<String, dynamic> json) =>
    IMetaGraphDef(
      metaInfoDef: json['metaInfoDef'] == null
          ? null
          : MetaGraphDef_IMetaInfoDef.fromJson(
              json['metaInfoDef'] as Map<String, dynamic>),
      graphDef: json['graphDef'] == null
          ? null
          : IGraphDef.fromJson(json['graphDef'] as Map<String, dynamic>),
      saverDef: json['saverDef'] == null
          ? null
          : ISaverDef.fromJson(json['saverDef'] as Map<String, dynamic>),
      collectionDef: (json['collectionDef'] as Map<String, dynamic>?)?.map(
        (k, e) =>
            MapEntry(k, ICollectionDef.fromJson(e as Map<String, dynamic>)),
      ),
      signatureDef: (json['signatureDef'] as Map<String, dynamic>?)?.map(
        (k, e) =>
            MapEntry(k, ISignatureDef.fromJson(e as Map<String, dynamic>)),
      ),
      assetFileDef: (json['assetFileDef'] as List<dynamic>?)
          ?.map((e) => IAssetFileDef.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$IMetaGraphDefToJson(IMetaGraphDef instance) =>
    <String, dynamic>{
      'metaInfoDef': instance.metaInfoDef,
      'graphDef': instance.graphDef,
      'saverDef': instance.saverDef,
      'collectionDef': instance.collectionDef,
      'signatureDef': instance.signatureDef,
      'assetFileDef': instance.assetFileDef,
    };

MetaGraphDef_IMetaInfoDef _$MetaGraphDef_IMetaInfoDefFromJson(
        Map<String, dynamic> json) =>
    MetaGraphDef_IMetaInfoDef(
      metaGraphVersion: json['metaGraphVersion'] as String?,
      strippedOpList: json['strippedOpList'] == null
          ? null
          : IOpList.fromJson(json['strippedOpList'] as Map<String, dynamic>),
      anyInfo: json['anyInfo'] == null
          ? null
          : IAny.fromJson(json['anyInfo'] as Map<String, dynamic>),
      tags: (json['tags'] as List<dynamic>?)?.map((e) => e as String).toList(),
      tensorflowVersion: json['tensorflowVersion'] as String?,
      tensorflowGitVersion: json['tensorflowGitVersion'] as String?,
    );

Map<String, dynamic> _$MetaGraphDef_IMetaInfoDefToJson(
        MetaGraphDef_IMetaInfoDef instance) =>
    <String, dynamic>{
      'metaGraphVersion': instance.metaGraphVersion,
      'strippedOpList': instance.strippedOpList,
      'anyInfo': instance.anyInfo,
      'tags': instance.tags,
      'tensorflowVersion': instance.tensorflowVersion,
      'tensorflowGitVersion': instance.tensorflowGitVersion,
    };

ISavedModel _$ISavedModelFromJson(Map<String, dynamic> json) => ISavedModel(
      savedModelSchemaVersion: json['savedModelSchemaVersion'],
      metaGraphs: (json['metaGraphs'] as List<dynamic>?)
          ?.map((e) => IMetaGraphDef.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$ISavedModelToJson(ISavedModel instance) =>
    <String, dynamic>{
      'savedModelSchemaVersion': instance.savedModelSchemaVersion,
      'metaGraphs': instance.metaGraphs,
    };

IFunctionDefLibrary _$IFunctionDefLibraryFromJson(Map<String, dynamic> json) =>
    IFunctionDefLibrary(
      function: (json['function'] as List<dynamic>?)
          ?.map((e) => IFunctionDef.fromJson(e as Map<String, dynamic>))
          .toList(),
      gradient: (json['gradient'] as List<dynamic>?)
          ?.map((e) => IGradientDef.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$IFunctionDefLibraryToJson(
        IFunctionDefLibrary instance) =>
    <String, dynamic>{
      'function': instance.function,
      'gradient': instance.gradient,
    };

IFunctionDef _$IFunctionDefFromJson(Map<String, dynamic> json) => IFunctionDef(
      signature: json['signature'] == null
          ? null
          : IOpDef.fromJson(json['signature'] as Map<String, dynamic>),
      attr: (json['attr'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, IAttrValue.fromJson(e as Map<String, dynamic>)),
      ),
      nodeDef: (json['nodeDef'] as List<dynamic>?)
          ?.map((e) => INodeDef.fromJson(e as Map<String, dynamic>))
          .toList(),
      ret: (json['ret'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, e as String),
      ),
    );

Map<String, dynamic> _$IFunctionDefToJson(IFunctionDef instance) =>
    <String, dynamic>{
      'signature': instance.signature,
      'attr': instance.attr,
      'nodeDef': instance.nodeDef,
      'ret': instance.ret,
    };

IGradientDef _$IGradientDefFromJson(Map<String, dynamic> json) => IGradientDef(
      functionName: json['functionName'] as String?,
      gradientFunc: json['gradientFunc'] as String?,
    );

Map<String, dynamic> _$IGradientDefToJson(IGradientDef instance) =>
    <String, dynamic>{
      'functionName': instance.functionName,
      'gradientFunc': instance.gradientFunc,
    };
