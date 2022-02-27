import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'Add',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'AddV2',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'AddN',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: 0,
        mapper: ParamMapper(
          name: 'tensors',
          type: 'tensors',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'BiasAdd',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Sub',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'RealDiv',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Div',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'DivNoNan',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'FloorDiv',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Mul',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Maximum',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Minimum',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Pow',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'SquaredDifference',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Mod',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'FloorMod',
    category: 'arithmetic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'a',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'b',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'T',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
];
