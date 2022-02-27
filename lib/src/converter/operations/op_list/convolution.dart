import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'AvgPool',
    category: 'convolution',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
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
      AttrParamMapper(
        tfName: 'ksize',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'kernelSize',
          type: 'number[]',
          defaultValue: null,
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
    tfOpName: 'MaxPool',
    category: 'convolution',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
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
      AttrParamMapper(
        tfName: 'ksize',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'kernelSize',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'explicit_paddings',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'explicitPaddings',
          type: 'number[]',
          defaultValue: [],
          notSupported: true,
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
    tfOpName: 'MaxPoolWithArgmax',
    category: 'convolution',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'ksize',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'kernelSize',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'include_batch_in_index',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'includeBatchInIndex',
          type: 'bool',
          defaultValue: null,
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
    tfOpName: 'AvgPool3D',
    category: 'convolution',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
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
      AttrParamMapper(
        tfName: 'ksize',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'kernelSize',
          type: 'number[]',
          defaultValue: null,
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
    tfOpName: 'MaxPool3D',
    category: 'convolution',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
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
      AttrParamMapper(
        tfName: 'ksize',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'kernelSize',
          type: 'number[]',
          defaultValue: null,
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
    tfOpName: 'Conv1D',
    category: 'convolution',
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
          name: 'filter',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'stride',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'stride',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: 'NWC',
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
      AttrParamMapper(
        tfName: 'dilation',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilation',
          type: 'number',
          defaultValue: 1,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Conv2D',
    category: 'convolution',
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
          name: 'filter',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'useCudnnOnGpu',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'useCudnnOnGpu',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: 'NHWC',
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'explicit_paddings',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'explicitPaddings',
          type: 'number[]',
          defaultValue: [],
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dilations',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: '_FusedConv2D',
    category: 'convolution',
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
          name: 'filter',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'explicit_paddings',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'explicitPaddings',
          type: 'number[]',
          defaultValue: [],
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'use_cudnn_on_gpu',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'useCudnnOnGpu',
          type: 'bool',
          defaultValue: true,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: 'NHWC',
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dilations',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: [1, 1, 1, 1],
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
        tfName: 'leakyrelu_alpha',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'leakyreluAlpha',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Conv2DBackpropInput',
    category: 'convolution',
    inputs: [
      InputParamMapper(
        start: 2,
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
          name: 'filter',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'outputShape',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
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
      AttrParamMapper(
        tfName: 'explicit_paddings',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'explicitPaddings',
          type: 'number[]',
          defaultValue: [],
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dilations',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'DepthwiseConv2d',
    category: 'convolution',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'input',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'filter',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: 'NHWC',
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'explicit_paddings',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'explicitPaddings',
          type: 'number[]',
          defaultValue: [],
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dilations',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'DepthwiseConv2dNative',
    category: 'convolution',
    inputs: [
      InputParamMapper(
        start: 0,
        end: null,
        mapper: ParamMapper(
          name: 'input',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      InputParamMapper(
        start: 1,
        end: null,
        mapper: ParamMapper(
          name: 'filter',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: 'NHWC',
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'explicit_paddings',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'explicitPaddings',
          type: 'number[]',
          defaultValue: [],
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dilations',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'FusedDepthwiseConv2dNative',
    category: 'convolution',
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
          name: 'filter',
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
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: 'NHWC',
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dilations',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: [1, 1, 1, 1],
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
        tfName: 'explicit_paddings',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'explicitPaddings',
          type: 'number[]',
          defaultValue: [],
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Conv3D',
    category: 'convolution',
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
          name: 'filter',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'data_format',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dataFormat',
          type: 'string',
          defaultValue: 'NHWC',
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'dilations',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'Dilation2D',
    category: 'convolution',
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
          name: 'filter',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'strides',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'strides',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'rates',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'dilations',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'padding',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'pad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
];
