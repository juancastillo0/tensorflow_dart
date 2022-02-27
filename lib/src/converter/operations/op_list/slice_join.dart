import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'ConcatV2',
    category: 'slice_join',
    inputs: [
      InputParamMapper(
        start: 0,
        end: -1,
        mapper: ParamMapper(
          name: 'tensors',
          type: 'tensors',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: -1,
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
        tfName: 'N',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'n',
          type: 'number',
          defaultValue: 2,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Concat',
    category: 'slice_join',
    inputs: [
      InputParamMapper(
        start: 1,
        end: 0,
        mapper: ParamMapper(
          name: 'tensors',
          type: 'tensors',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 0,
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
        tfName: 'N',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'n',
          type: 'number',
          defaultValue: 2,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'GatherV2',
    category: 'slice_join',
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
          name: 'indices',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'axis',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'batch_dims',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'batchDims',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Gather',
    category: 'slice_join',
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
          name: 'indices',
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
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Reverse',
    category: 'slice_join',
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
          name: 'dims',
          type: 'bool[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'ReverseV2',
    category: 'slice_join',
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
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Slice',
    category: 'slice_join',
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
          name: 'begin',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'size',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'StridedSlice',
    category: 'slice_join',
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
          name: 'begin',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'end',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'begin_mask',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'beginMask',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'end_mask',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'endMask',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'new_axis_mask',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'newAxisMask',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'ellipsis_mask',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'ellipsisMask',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'shrink_axis_mask',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'shrinkAxisMask',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Pack',
    category: 'slice_join',
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
        tfName: 'axis',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'axis',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Unpack',
    category: 'slice_join',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensor',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'axis',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'axis',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'num',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'num',
          type: 'number',
          defaultValue: 0,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Tile',
    category: 'slice_join',
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
          name: 'reps',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Split',
    category: 'slice_join',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'axis',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
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
        tfName: 'num_split',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'numOrSizeSplits',
          type: 'number',
          defaultValue: 1,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'SplitV',
    category: 'slice_join',
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
          name: 'numOrSizeSplits',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'axis',
          type: 'number',
          defaultValue: 0,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'ScatterNd',
    category: 'slice_join',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'indices',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'values',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'shape',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'GatherNd',
    category: 'slice_join',
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
          name: 'indices',
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
    category: 'slice_join',
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
          defaultValue: false,
          notSupported: true,
        ),
      ),
    ],
  ),
];
