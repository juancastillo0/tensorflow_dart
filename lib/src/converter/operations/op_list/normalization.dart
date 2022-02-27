import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'FusedBatchNorm',
    category: 'normalization',
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
          name: 'scale',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'offset',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'mean',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 4,
        end: null,
        mapper: ParamMapper(
          name: 'variance',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'epsilon',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'epsilon',
          type: 'number',
          defaultValue: 0.001,
          notSupported: null,
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
    tfOpName: 'FusedBatchNormV2',
    category: 'normalization',
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
          name: 'scale',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'offset',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'mean',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 4,
        end: null,
        mapper: ParamMapper(
          name: 'variance',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'epsilon',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'epsilon',
          type: 'number',
          defaultValue: 0.001,
          notSupported: null,
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
    tfOpName: 'FusedBatchNormV3',
    category: 'normalization',
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
          name: 'scale',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'offset',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'mean',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 4,
        end: null,
        mapper: ParamMapper(
          name: 'variance',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'epsilon',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'epsilon',
          type: 'number',
          defaultValue: 0.001,
          notSupported: null,
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
    tfOpName: 'LRN',
    category: 'normalization',
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
        tfName: 'depth_radius',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'radius',
          type: 'number',
          defaultValue: 5,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'bias',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'bias',
          type: 'number',
          defaultValue: 1,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'alpha',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'alpha',
          type: 'number',
          defaultValue: 1,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'beta',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'beta',
          type: 'number',
          defaultValue: 0.5,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Softmax',
    category: 'normalization',
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
    tfOpName: 'LogSoftmax',
    category: 'normalization',
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
    tfOpName: 'SparseToDense',
    category: 'normalization',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'sparseIndices',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'outputShape',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'sparseValues',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
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
        tfName: 'validate_indices',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'validateIndices',
          type: 'bool',
          defaultValue: true,
          notSupported: true,
        ),
      ),
    ],
  ),
];
