import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'EmptyTensorList',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'maxNumElements',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'LoopCond',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'pred',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Switch',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'data',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'pred',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'Merge',
    category: 'control',
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
    tfOpName: 'Enter',
    category: 'control',
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
        tfName: 'frame_name',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'frameName',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'is_constant',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'isConstant',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Exit',
    category: 'control',
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
    tfOpName: 'NextIteration',
    category: 'control',
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
    tfOpName: 'TensorArrayV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'size',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
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
      AttrParamMapper(
        tfName: 'element_shape',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dynamic_size',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dynamicSize',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'clear_after_read',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'clearAfterRead',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'identical_element_shapes',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'identicalElementShapes',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'tensor_array_name',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'name',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorArrayWriteV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'index',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'tensor',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'flowIn',
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
    ],
  ),
  OpMapper(
    tfOpName: 'TensorArrayReadV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'index',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'flowIn',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'dtype',
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
    tfOpName: 'TensorArrayGatherV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'flowIn',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
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
      AttrParamMapper(
        tfName: 'element_shape',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorArrayScatterV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'tensor',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'flowIn',
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
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorArrayConcatV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'flowIn',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
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
      AttrParamMapper(
        tfName: 'element_shape_except0',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementShapeExcept0',
          type: 'shape',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorArraySplitV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'tensor',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'lengths',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'flowIn',
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
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorArraySizeV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'flowIn',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'TensorArrayCloseV3',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorArrayId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'StatelessIf',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'cond',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
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
        tfName: 'then_branch',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'thenBranch',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'else_branch',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elseBranch',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'If',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'cond',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
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
        tfName: 'then_branch',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'thenBranch',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'else_branch',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elseBranch',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'StatelessWhile',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
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
        tfName: 'cond',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'cond',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'body',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'body',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'While',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
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
        tfName: 'cond',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'cond',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'body',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'body',
          type: 'func',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListScatter',
    category: 'control',
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
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'indices',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListScatterV2',
    category: 'control',
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
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'indices',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 3,
        end: null,
        mapper: ParamMapper(
          name: 'numElements',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListGather',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorListId',
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
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListGetItem',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorListId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'index',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListSetItem',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorListId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'index',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
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
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListReserve',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'numElements',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListFromTensor',
    category: 'control',
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
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListStack',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorListId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'num_elements',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'numElements',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListSplit',
    category: 'control',
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
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 2,
        end: null,
        mapper: ParamMapper(
          name: 'lengths',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListConcat',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorListId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_shape',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListPopBack',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorListId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'elementShape',
          type: 'shape',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'TensorListPushBack',
    category: 'control',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'tensorListId',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
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
        tfName: 'element_dtype',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'elementDType',
          type: 'dtype',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
];
