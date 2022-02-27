import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'NonMaxSuppressionV2',
    category: 'dynamic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'boxes',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'scores',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'maxOutputSize',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'iouThreshold',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'NonMaxSuppressionV3',
    category: 'dynamic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'boxes',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'scores',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'maxOutputSize',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'iouThreshold',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 4,
        end: null,
        mapper: ParamMapper(
          name: 'scoreThreshold',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'NonMaxSuppressionV4',
    category: 'dynamic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'boxes',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'scores',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'maxOutputSize',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'iouThreshold',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 4,
        end: null,
        mapper: ParamMapper(
          name: 'scoreThreshold',
          type: 'number',
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
        tfName: 'T_threshold',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'threshold',
          type: 'dtype',
          defaultValue: null,
          notSupported: true,
        ),
      ),
      AttrParamMapper(
        tfName: 'pad_to_max_output_size',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'padToMaxOutputSize',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'NonMaxSuppressionV5',
    category: 'dynamic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'boxes',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'scores',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'maxOutputSize',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'iouThreshold',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 4,
        end: null,
        mapper: ParamMapper(
          name: 'scoreThreshold',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 5,
        end: null,
        mapper: ParamMapper(
          name: 'softNmsSigma',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Where',
    category: 'dynamic',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'condition',
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
    tfOpName: 'ListDiff',
    category: 'dynamic',
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
          name: 'y',
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
