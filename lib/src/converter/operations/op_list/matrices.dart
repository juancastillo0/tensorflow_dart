import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: '_FusedMatMul',
    category: 'matrices',
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
      InputParamMapper(
        start: 2,
        end: 0,
        mapper: ParamMapper(
          name: 'args',
          type: 'tensors',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'num_args',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'numArgs',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'fused_ops',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'fusedOps',
          type: 'string[]',
          defaultValue: [],
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'epsilon',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'epsilon',
          type: 'number',
          defaultValue: 0.0001,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'transpose_a',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeA',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'transpose_b',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeB',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
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
    tfOpName: 'MatMul',
    category: 'matrices',
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
        tfName: 'transpose_a',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeA',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'transpose_b',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeB',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
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
    tfOpName: 'BatchMatMul',
    category: 'matrices',
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
        tfName: 'adj_x',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeA',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'adj_y',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeB',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
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
    tfOpName: 'BatchMatMulV2',
    category: 'matrices',
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
        tfName: 'adj_x',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeA',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'adj_y',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'transposeB',
          type: 'bool',
          defaultValue: false,
          notSupported: null,
        ),
      ),
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
    tfOpName: 'Transpose',
    category: 'matrices',
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
          name: 'perm',
          type: 'number[]',
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
    tfOpName: 'Einsum',
    category: 'matrices',
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
    attrs: [
      AttrParamMapper(
        tfName: 'equation',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'equation',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'N',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'n',
          type: 'number',
          defaultValue: 2,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'T',
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
];
