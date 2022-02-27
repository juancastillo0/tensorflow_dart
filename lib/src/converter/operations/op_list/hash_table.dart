import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'HashTable',
    category: 'hash_table',
    inputs: [],
    attrs: [
      AttrParamMapper(
        tfName: 'shared_name',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'sharedName',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'use_node_name_sharing',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'useNodeNameSharing',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'key_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keyDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'value_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'valueDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'HashTableV2',
    category: 'hash_table',
    inputs: [],
    attrs: [
      AttrParamMapper(
        tfName: 'shared_name',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'sharedName',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'use_node_name_sharing',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'useNodeNameSharing',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'key_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keyDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'value_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'valueDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'LookupTableImport',
    category: 'hash_table',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tableHandle',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'keys',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'values',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'Tin',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tIn',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
      AttrParamMapper(
        tfName: 'Tout',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tOut',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'LookupTableImportV2',
    category: 'hash_table',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tableHandle',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'keys',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'values',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'Tin',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tIn',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
      AttrParamMapper(
        tfName: 'Tout',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tOut',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'LookupTableFind',
    category: 'hash_table',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tableHandle',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'keys',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'defaultValue',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'Tin',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tIn',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
      AttrParamMapper(
        tfName: 'Tout',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tOut',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'LookupTableFindV2',
    category: 'hash_table',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tableHandle',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'keys',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'defaultValue',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'Tin',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tIn',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
      AttrParamMapper(
        tfName: 'Tout',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'tOut',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'LookupTableSize',
    category: 'hash_table',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tableHandle',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'LookupTableSizeV2',
    category: 'hash_table',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tableHandle',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
];
