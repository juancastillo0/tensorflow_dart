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

/**
 * Tokenizer.encode() is a port of `EncodeAsIds` from the SentencePiece library
 * (https://github.com/google/sentencepiece). Encode uses the Viterbi algorithm
 * to find the most likely sequence of tokens that comprise the input. For more
 * details, refer to https://arxiv.org/pdf/1804.10959.pdf.
 */

// import * as tf from '@tensorflow/tfjs-core';

// import {stringToChars} from '../util';

// import {Trie} from './trie';

import 'dart:convert';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:http/http.dart' as http;
import "package:unorm_dart/unorm_dart.dart" as unorm;

import '../util.dart';
import 'trie.dart';

const separator =
    '\u2581'; // This is the unicode character 'lower one eighth block'.

String _processInput(String str) {
  final normalized = unorm.nfkc(str);
  return normalized.length > 0
      ? separator + normalized.replaceAll(RegExp(' '), separator)
      : normalized;
}

// The first tokens are reserved for unk, control symbols, and user-defined
// symbols.
const RESERVED_SYMBOLS_COUNT = 6;

typedef Vocabulary = List<MapEntry<String, int>>;

class Score {
  final List<String> key;
  final int score;
  final int index;

  Score({
    required this.key,
    required this.score,
    required this.index,
  });
}

class Tokenizer {
  final Trie trie = Trie();
  final Vocabulary vocabulary;
  final int reservedSymbolsCount;

  Tokenizer(
    this.vocabulary, {
    this.reservedSymbolsCount = RESERVED_SYMBOLS_COUNT,
  }) {
    for (int i = this.reservedSymbolsCount; i < this.vocabulary.length; i++) {
      this.trie.insert(this.vocabulary[i].key, this.vocabulary[i].value, i);
    }
  }

  List<int> encode(String input) {
    final List<Map<int, List<Score>>> nodes = [];
    final List<int> words = [];
    final List<int> best = [];

    input = _processInput(input);

    final symbols = stringToChars(input);

    for (int i = 0; i <= symbols.length; i++) {
      nodes.add({});
      words.add(0);
      best.add(0);
    }

    // Construct the lattice.
    for (int i = 0; i < symbols.length; i++) {
      final matches = this.trie.commonPrefixSearch(symbols.slice(i));

      for (int j = 0; j < matches.length; j++) {
        final piece = matches[j];
        final obj = Score(
          key: piece.token,
          score: piece.score,
          index: piece.index,
        );

        final endPos = piece.token.length;
        if (nodes[i + endPos][i] == null) {
          nodes[i + endPos][i] = [];
        }

        nodes[i + endPos][i]!.add(obj);
      }
    }

    for (int endPos = 0; endPos <= symbols.length; endPos++) {
      for (final arr in nodes[endPos].values) {
        for (int j = 0; j < arr.length; j++) {
          final word = arr[j];
          final score = word.score + best[endPos - word.key.length];

          if (best[endPos] == 0 || score >= best[endPos]) {
            best[endPos] = score;
            words[endPos] = arr[j].index;
          }
        }
      }
    }

    final List<int> results = [];

    // Backward pass.
    var iter = words.length - 1;
    while (iter > 0) {
      results.add(words[iter]);
      iter -= this.vocabulary[words[iter]].key.length;
    }

    // Merge consecutive unks.
    final merged = <int>[];
    var isPreviousUnk = false;
    for (int i = 0; i < results.length; i++) {
      final id = results[i];
      if (!(isPreviousUnk && id == 0)) {
        merged.add(id);
      }

      isPreviousUnk = id == 0;
    }

    return merged.reversed.toList();
  }
}

/**
 * Load the Tokenizer for use independently from the UniversalSentenceEncoder.
 *
 * @param pathToVocabulary (optional) Provide a path to the vocabulary file.
 */
Future<Tokenizer> loadTokenizer(String pathToVocabulary) async {
  final vocabulary = await loadVocabulary(pathToVocabulary);
  final tokenizer = Tokenizer(vocabulary);
  return tokenizer;
}

/**
 * Load a vocabulary for the Tokenizer.
 *
 * @param pathToVocabulary Defaults to the path to the 8k vocabulary used by the
 * UniversalSentenceEncoder.
 */
Future<Vocabulary> loadVocabulary(String pathToVocabulary) async {
  final stream =
      await tf.env().platform!.fetch(Uri.parse(pathToVocabulary), null);
  final response = await http.Response.fromStream(stream);
  return (jsonDecode(response.body) as List)
      .map((e) => MapEntry(e[0] as String, e[1] as int))
      .toList();
}
