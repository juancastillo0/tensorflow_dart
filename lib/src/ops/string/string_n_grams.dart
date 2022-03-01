/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import {ENGINE} from '../../engine';
// import {StringNGrams, StringNGramsAttrs, StringNGramsInputs} from '../../kernel_names';
// import {Tensor, Tensor1D} from '../../tensor';
// import {NamedTensorMap} from '../../tensor_types';
// import {convertToTensor} from '../../tensor_util_env';
// import {TensorLike} from '../../types';
// import {op} from '../operation';

import '../_prelude.dart';

/**
 * Creates ngrams from ragged string data.
 *
 * This op accepts a ragged tensor with 1 ragged dimension containing only
 * strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
 * of that string, joined along the innermost axis.
 *
 * ```js
 * const result = tf.string.stringNGrams(
 *   ['a', 'b', 'c', 'd'], tf.tensor1d([0, 2, 4], 'int32'),
 *   '|', [1, 2], 'LP', 'RP', -1, false);
 * result['nGrams'].print(); // ['a', 'b', 'LP|a', 'a|b', 'b|RP',
 *                           //  'c', 'd', 'LP|c', 'c|d', 'd|RP']
 * result['nGramsSplits'].print(); // [0, 5, 10]
 * ```
 * @param data: The values tensor of the ragged string tensor to make ngrams out
 *     of. Must be a 1D string tensor.
 * @param dataSplits: The splits tensor of the ragged string tensor to make
 *     ngrams out of.
 * @param separator: The string to append between elements of the token. Use ""
 *     for no separator.
 * @param nGramWidths: The sizes of the ngrams to create.
 * @param leftPad: The string to use to pad the left side of the ngram sequence.
 *     Only used if pad_width !== 0.
 * @param rightPad: The string to use to pad the right side of the ngram
 *     sequence. Only used if pad_width !== 0.
 * @param padWidth: The number of padding elements to add to each side of each
 *     sequence. Note that padding will never be greater than `nGramWidths`-1
 *     regardless of this value. If `padWidth`=-1 , then add max(`nGramWidths)-1
 *     elements.
 * @param preserveShortSequences: If true, then ensure that at least one ngram
 *     is generated for each input sequence. In particular, if an input sequence
 *     is shorter than min(ngramWidth) + 2*padWidth, then generate a single
 *     ngram containing the entire sequence. If false, then no ngrams are
 *     generated for these short input sequences.
 * @return A map with the following properties:
 *     - nGrams: The values tensor of the output ngrams ragged tensor.
 *     - nGramsSplits: The splits tensor of the output ngrams ragged tensor.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
StringNGramsResult stringNGrams(
  Tensor1D data,
  Tensor dataSplits,
  String separator,
  List<int> nGramWidths,
  String leftPad,
  String rightPad,
  int padWidth,
  bool preserveShortSequences,
) {
  return execOp('stringNGrams', () {
    final $data = convertToTensor(data, 'data', 'stringNGrams', 'string');
    if ($data.dtype != 'string') {
      throw Exception('Data must be of datatype string');
    }
    if ($data.shape.length != 1) {
      throw Exception('Data must be a vector, saw: ${$data.shape}');
    }

    final $dataSplits =
        convertToTensor(dataSplits, 'dataSplits', 'stringNGrams');
    if ($dataSplits.dtype != 'int32') {
      throw Exception('Data splits must be of datatype int32');
    }

    final attrs = {
      // StringNGramsAttrs
      'separator': separator,
      'nGramWidths': nGramWidths,
      'leftPad': leftPad,
      'rightPad': rightPad,
      'padWidth': padWidth,
      'preserveShortSequences': preserveShortSequences
    };

    final inputs = {
      'data': $data,
      'dataSplits': $dataSplits
    }; // StringNGramsInputs
    final result =
        ENGINE.runKernel(StringNGrams, inputs, attrs) as List<Tensor>;
    return StringNGramsResult(nGrams: result[0], nGramsSplits: result[1]);
  });
}

class StringNGramsResult {
  final Tensor nGrams;
  final Tensor nGramsSplits;

  StringNGramsResult({
    required this.nGrams,
    required this.nGramsSplits,
  });
}
