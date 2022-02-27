import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'TopKV2',
    category: 'evaluation',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'x',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'k',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'sorted',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'sorted',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Unique',
    category: 'evaluation',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'x',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'UniqueV2',
    category: 'evaluation',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'x',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'axis',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
];
