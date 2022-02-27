import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'StringNGrams',
    category: 'string',
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
          name: 'dataSplits',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'separator',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'separator',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'ngram_widths',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'nGramWidths',
          type: 'number[]',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'left_pad',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'leftPad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'right_pad',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'rightPad',
          type: 'string',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'pad_width',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'padWidth',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
      AttrParamMapper(
        tfName: 'preserve_short_sequences',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'preserveShortSequences',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'StringSplit',
    category: 'string',
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
          name: 'delimiter',
          type: 'tensor',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'skip_empty',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'skipEmpty',
          type: 'bool',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
  OpMapper(
    tfOpName: 'StringToHashBucketFast',
    category: 'string',
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
    ],
    attrs: [
      AttrParamMapper(
        tfName: 'num_buckets',
        tfDeprecatedName: null,
        mapper: ParamMapper(
          name: 'numBuckets',
          type: 'number',
          defaultValue: null,
          notSupported: null,
        ),
      ),
    ],
  ),
];
