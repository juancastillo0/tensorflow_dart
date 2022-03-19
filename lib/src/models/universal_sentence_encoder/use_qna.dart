/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import * as tfconv from '@tensorflow/tfjs-converter';
// import * as tf from '@tensorflow/tfjs-core';

// import {loadVocabulary, Tokenizer} from './tokenizer';

// export {version} from './version';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;

import 'tokenizer/tokenizer.dart';

const BASE_PATH =
    'https://tfhub.dev/google/tfjs-model/universal-sentence-encoder-qa-ondevice/1';
// Index in the vocab file that needs to be skipped.
const SKIP_VALUES = [0, 1, 2];
// Offset value for skipped vocab index.
const OFFSET = 3;
// Input tensor size limit.
const INPUT_LIMIT = 192;
// Model node name for query.
const QUERY_NODE_NAME = 'input_inp_text';
// Model node name for query.
const RESPONSE_CONTEXT_NODE_NAME = 'input_res_context';
// Model node name for response.
const RESPONSE_NODE_NAME = 'input_res_text';
// Model node name for response result.
const RESPONSE_RESULT_NODE_NAME = 'Final/EncodeResult/mul';
// Model node name for query result.
const QUERY_RESULT_NODE_NAME = 'Final/EncodeQuery/mul';
// Reserved symbol count for tokenizer.
const RESERVED_SYMBOLS_COUNT = 3;
// Value for token padding
const TOKEN_PADDING = 2;
// Start value for each token
const TOKEN_START_VALUE = 1;

class ModelOutput {
  final tf.Tensor queryEmbedding;
  final tf.Tensor responseEmbedding;

  const ModelOutput({
    required this.queryEmbedding,
    required this.responseEmbedding,
  });
}

class ModelInput {
  final List<String> queries;
  final List<String> responses;
  final List<String>? contexts;

  const ModelInput({
    required this.queries,
    required this.responses,
    this.contexts,
  });
}

Future<UniversalSentenceEncoderQnA> loadQnA() async {
  final use = UniversalSentenceEncoderQnA();
  await use.load();
  return use;
}

class UniversalSentenceEncoderQnA {
  late final tfconv.GraphModel model;
  late final Tokenizer tokenizer;

  Future<tfconv.GraphModel> loadModel() {
    return tfconv.loadGraphModel(
      tfconv.ModelHandler.fromUrl(BASE_PATH),
      tfconv.LoadOptions(fromTFHub: true),
    );
  }

  Future<void> load() async {
    final _l = await Future.wait([
      this.loadModel(),
      loadVocabulary('${BASE_PATH}/vocab.json?tfjs-format=file'),
    ]);

    this.model = _l[0] as tfconv.GraphModel;

    final vocabulary = _l[1] as Vocabulary;
    this.tokenizer =
        Tokenizer(vocabulary, reservedSymbolsCount: RESERVED_SYMBOLS_COUNT);
  }

  /**
   *
   * Returns a map of queryEmbedding and responseEmbedding
   *
   * @param input the ModelInput that contains queries and answers.
   */
  ModelOutput embed(ModelInput input) {
    final embeddings = tf.tidy(() {
      final queryEncoding = this._tokenizeStrings(input.queries, INPUT_LIMIT);
      final responseEncoding =
          this._tokenizeStrings(input.responses, INPUT_LIMIT);
      if (input.contexts != null) {
        if (input.contexts!.length != input.responses.length) {
          throw Exception('The length of response strings ' +
              'and context strings need to match.');
        }
      }
      final contexts = input.contexts ?? [];
      if (input.contexts == null) {
        contexts.length = input.responses.length;
        contexts.fillRange(0, contexts.length, '');
      }
      final contextEncoding = this._tokenizeStrings(contexts, INPUT_LIMIT);
      final modelInputs = tf.TensorMap({
        QUERY_NODE_NAME: queryEncoding,
        RESPONSE_NODE_NAME: responseEncoding,
        RESPONSE_CONTEXT_NODE_NAME: contextEncoding,
      });

      return this.model.execute(
          modelInputs, [QUERY_RESULT_NODE_NAME, RESPONSE_RESULT_NODE_NAME]);
    }) as List<tf.Tensor>;
    final queryEmbedding = embeddings[0];
    final responseEmbedding = embeddings[1];

    return ModelOutput(
      queryEmbedding: queryEmbedding,
      responseEmbedding: responseEmbedding,
    );
  }

  tf.Tensor2D _tokenizeStrings(List<String> strs, int limit) {
    final tokens = strs
        .map((s) => this._shiftTokens(this.tokenizer.encode(s), INPUT_LIMIT))
        .toList();
    return tf.tensor2d(tokens, [strs.length, INPUT_LIMIT], 'int32');
  }

  List<int> _shiftTokens(List<int> tokens, int limit) {
    tokens.insert(0, TOKEN_START_VALUE);
    for (int index = 0; index < limit; index++) {
      if (index >= tokens.length) {
        tokens[index] = TOKEN_PADDING;
      } else if (!SKIP_VALUES.contains(tokens[index])) {
        tokens[index] += OFFSET;
      }
    }
    return tokens.slice(0, limit);
  }
}
