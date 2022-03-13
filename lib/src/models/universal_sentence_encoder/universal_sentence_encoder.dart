/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// import * as tfconv from '@tensorflow/tfjs-converter';
// import * as tf from '@tensorflow/tfjs-core';

// import {loadTokenizer, loadVocabulary, Tokenizer} from './tokenizer';
// import {loadQnA} from './use_qna';

// export {version} from './version';

import 'package:collection/collection.dart';
import 'tokenizer/tokenizer.dart';

import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;
import 'package:tensorflow_wasm/src/util_base.dart' as util;

export 'tokenizer/tokenizer.dart' show loadTokenizer, Tokenizer;
export 'use_qna.dart' show loadQnA;

const BASE_PATH =
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder';

// class ModelInputs extends tf.NamedTensorMap {
//   indices: tf.Tensor;
//   values: tf.Tensor;
// }

class LoadConfig {
  final String? modelUrl;
  final String? vocabUrl;

  LoadConfig({
    this.modelUrl,
    this.vocabUrl,
  });
}

Future<UniversalSentenceEncoder> load([LoadConfig? config]) async {
  final use = UniversalSentenceEncoder();
  await use.load(config);
  return use;
}

class UniversalSentenceEncoder {
  late final tfconv.GraphModel model;
  late final Tokenizer tokenizer;

  Future<tfconv.GraphModel> loadModel([String? modelUrl]) {
    return modelUrl != null
        ? tfconv.loadGraphModel(tfconv.ModelHandler.fromUrl(modelUrl))
        : tfconv.loadGraphModel(
            tfconv.ModelHandler.fromUrl(
                'https://tfhub.dev/tensorflow/tfjs-model/universal-sentence-encoder-lite/1/default/1'),
            tfconv.LoadOptions(fromTFHub: true));
  }

  Future<void> load([LoadConfig? config]) async {
    final _l = await Future.wait([
      this.loadModel(config?.modelUrl),
      loadVocabulary(config?.vocabUrl ?? '${BASE_PATH}/vocab.json')
    ]);

    this.model = _l[0] as tfconv.GraphModel;
    this.tokenizer = Tokenizer(_l[1] as Vocabulary);
  }

  /**
   *
   * Returns a 2D Tensor of shape [input.length, 512] that contains the
   * Universal Sentence Encoder embeddings for each input.
   *
   * @param inputs A string or an array of strings to embed.
   */
  Future<tf.Tensor2D> embed(List<String> inputs) async {
    // if (typeof inputs == 'string') {
    //   inputs = [inputs];
    // }

    final encodings = inputs.map((d) => this.tokenizer.encode(d)).toList();

    final flattenedIndicesArr = encodings
        .expandIndexed((i, arr) => arr.mapIndexed((index, d) => [i, index]))
        .toList();

    // List<List<int>> flattenedIndicesArr = [];
    // for (int i = 0; i < indicesArr.length; i++) {
    //   flattenedIndicesArr =
    //       flattenedIndicesArr.concat(indicesArr[i]);
    // }

    final indices = tf.tensor2d(
        flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
    final values = tf.tensor1d(util.flatten(encodings), 'int32');

    final modelInputs = tf.TensorMap({'indices': indices, 'values': values});

    final embeddings = await this.model.executeAsync(modelInputs);
    indices.dispose();
    values.dispose();

    return embeddings as tf.Tensor2D;
  }
}
