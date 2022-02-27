import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'PlaceholderWithDefault',
    category: 'graph',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'default',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'shape',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'shape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Placeholder',
    category: 'graph',
    inputs: [],
    attrs: [
      AttrParamMapper(
        tfName: 'shape',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'shape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dtype',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Const',
    category: 'graph',
    inputs: [],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Identity',
    category: 'graph',
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
    tfOpName: 'IdentityN',
    category: 'graph',
    inputs: [
      InputParamMapper(
        start: 0,
        end: 0,
        mapper: ParamMapper(
          name: 'x',
          type: 'tensors',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Snapshot',
    category: 'graph',
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
    tfOpName: 'Rank',
    category: 'graph',
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
    tfOpName: 'Size',
    category: 'graph',
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
    tfOpName: 'Shape',
    category: 'graph',
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
    tfOpName: 'ShapeN',
    category: 'graph',
    inputs: [
      InputParamMapper(
        start: 0,
        end: 0,
        mapper: ParamMapper(
          name: 'x',
          type: 'tensors',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Print',
    category: 'graph',
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
          name: 'data',
          type: 'tensors',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'message',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'message',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'first_n',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'firstN',
          type: 'number',
          defaultValue: null,
          notSupported: true,
        ),
      ),
      AttrParamMapper(
        tfName: 'summarize',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'summarize',
          type: 'number',
          defaultValue: 3,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'NoOp',
    category: 'graph',
    inputs: [],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'StopGradient',
    category: 'graph',
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
    tfOpName: 'FakeQuantWithMinMaxVars',
    category: 'graph',
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
    attrs: [
      AttrParamMapper(
        tfName: 'min',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'min',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'max',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'max',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
];
