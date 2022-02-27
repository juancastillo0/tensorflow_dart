import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'Bincount',
    category: 'reduction',
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
          name: 'size',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'weights',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'DenseBincount',
    category: 'reduction',
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
          name: 'size',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'weights',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'binary_output',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'binaryOutput',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Max',
    category: 'reduction',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'keep_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keepDims',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Mean',
    category: 'reduction',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'keep_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keepDims',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Min',
    category: 'reduction',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'keep_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keepDims',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Sum',
    category: 'reduction',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'keep_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keepDims',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'All',
    category: 'reduction',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'keep_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keepDims',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Any',
    category: 'reduction',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'keep_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keepDims',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'ArgMax',
    category: 'reduction',
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
  OpMapper(
    tfOpName: 'ArgMin',
    category: 'reduction',
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
  OpMapper(
    tfOpName: 'Prod',
    category: 'reduction',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'keep_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'keepDims',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Cumsum',
    category: 'reduction',
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
    attrs: [
      AttrParamMapper(
        tfName: 'exclusive',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'exclusive',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'reverse',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'reverse',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
];
