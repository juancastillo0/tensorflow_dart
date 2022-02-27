import '../types.dart';

const opMappers = [
  OpMapper(
    tfOpName: 'FFT',
    category: 'spectral',
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
    tfOpName: 'IFFT',
    category: 'spectral',
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
    tfOpName: 'RFFT',
    category: 'spectral',
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
          name: 'fft_length',
          type: 'number',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
    attrs: [],
  ),
  OpMapper(
    tfOpName: 'IRFFT',
    category: 'spectral',
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
          name: 'fft_length',
          type: 'number',
          defaultValue: null,
          notSupported: true,
        ),
      ),
    ],
    attrs: [],
  ),
];
