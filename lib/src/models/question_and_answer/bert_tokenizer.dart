/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import 'dart:convert';

import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import "package:unorm_dart/unorm_dart.dart" as unorm;

const SEPERATOR = '\u2581';
const UNK_INDEX = 100;
const CLS_INDEX = 101;
const CLS_TOKEN = '[CLS]';
const SEP_INDEX = 102;
const SEP_TOKEN = '[SEP]';
const VOCAB_BASE = 'https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1/';
const VOCAB_URL = VOCAB_BASE + 'processed_vocab.json?tfjs-format=file';

/**
 * Class for represent node for token parsing Trie data structure.
 */
class TrieNode {
  final TrieNode? parent;
  final Map<String, TrieNode> children = {};
  bool end = false;

  int? score;
  int? index;
  final String? key;

  TrieNode(this.key, [this.parent]);

  Word getWord() {
    final List<String> output = [];
    TrieNode? node = this;

    while (node != null) {
      if (node.key != null) {
        output.insert(0, node.key!);
      }
      node = node.parent;
    }

    return Word(
      output: output,
      score: this.score,
      index: this.index,
    );
  }
}

class Word {
  final List<String> output;
  final int? score;
  final int? index;

  Word({
    required this.output,
    required this.score,
    required this.index,
  });
}

class Trie {
  final root = TrieNode(null);

  /**
   * Insert the bert vacabulary word into the trie.
   * @param word word to be inserted.
   * @param score word score.
   * @param index index of word in the bert vocabulary file.
   */
  insert(String word, int score, int index) {
    var node = this.root;

    final symbols = <String>[];
    for (final _ch in word.codeUnits) {
      final symbol = String.fromCharCode(_ch);
      symbols.add(symbol);
    }

    for (int i = 0; i < symbols.length; i++) {
      if (node.children[symbols[i]] == null) {
        final t = TrieNode(symbols[i], node);
        node.children[symbols[i]] = t;
      }

      node = node.children[symbols[i]]!;

      if (i == symbols.length - 1) {
        node.end = true;
        node.score = score;
        node.index = index;
      }
    }
  }

  /**
   * Find the Trie node for the given token, it will return the first node that
   * matches the subtoken from the beginning of the token.
   * @param token string, input string to be searched.
   */
  TrieNode? find(String token) {
    TrieNode? node = this.root;
    int iter = 0;

    while (iter < token.length && node != null) {
      node = node.children[token[iter]];
      iter++;
    }

    return node;
  }
}

bool _isWhitespace(String ch) {
  return RegExp(r'\s').hasMatch(ch);
}

bool _isInvalid(String ch) {
  return (ch.codeUnitAt(0) == 0 || ch.codeUnitAt(0) == 0xfffd);
}

final punctuations = '[~`!@#\$%^&*(){}[];:"\'<,.>?/\\|-_+=';

/** To judge whether it's a punctuation. */
bool _isPunctuation(String ch) {
  return punctuations.indexOf(ch) != -1;
}

class Token {
  String _text;
  final int index;

  String get text => _text;

  Token({
    required String text,
    required this.index,
  }) : _text = text;
}

/**
 * Tokenizer for Bert.
 */
class BertTokenizer {
  late final List<String> vocab;
  late final Trie trie;

  /**
   * Load the vacabulary file and initialize the Trie for lookup.
   */
  Future<void> load() async {
    this.vocab = await this._loadVocab();

    this.trie = Trie();
    // Actual tokens start at 999.
    for (int vocabIndex = 999; vocabIndex < this.vocab.length; vocabIndex++) {
      final word = this.vocab[vocabIndex];
      this.trie.insert(word, 1, vocabIndex);
    }
  }

  Future<List<String>> _loadVocab() async {
    return tf
        .env()
        .platform!
        .fetchAndParse(Uri.parse(VOCAB_URL))
        .then((d) => (jsonDecode(d.body) as List).cast<String>());
  }

  List<Token> processInput(String text) {
    final List<int> charOriginalIndex = [];
    final cleanedText = this._cleanText(text, charOriginalIndex);
    final origTokens = cleanedText.split(' ');

    int charCount = 0;
    final tokens = origTokens.expand((token) {
      token = token.toLowerCase();
      final tokens = this._runSplitOnPunc(token, charCount, charOriginalIndex);
      charCount += token.length + 1;
      return tokens;
    }).toList();
    return tokens;
  }

  /* Performs invalid character removal and whitespace cleanup on text. */
  String _cleanText(String text, List<int> charOriginalIndex) {
    final List<String> stringBuilder = [];
    int originalCharIndex = 0;
    for (final _ch in text.codeUnits) {
      final ch = String.fromCharCode(_ch);
      // Skip the characters that cannot be used.
      if (_isInvalid(ch)) {
        originalCharIndex += ch.length;
        continue;
      }
      if (_isWhitespace(ch)) {
        if (stringBuilder.length > 0 &&
            stringBuilder[stringBuilder.length - 1] != ' ') {
          stringBuilder.add(' ');
          charOriginalIndex.add(originalCharIndex);
          originalCharIndex += ch.length;
        } else {
          originalCharIndex += ch.length;
          continue;
        }
      } else {
        stringBuilder.add(ch);
        charOriginalIndex.add(originalCharIndex);
        originalCharIndex += ch.length;
      }
    }
    return stringBuilder.join('');
  }

  /* Splits punctuation on a piece of text. */
  List<Token> _runSplitOnPunc(
    String text,
    int count,
    List<int> charOriginalIndex,
  ) {
    final List<Token> tokens = [];
    bool startNewWord = true;
    for (final _ch in text.codeUnits) {
      final ch = String.fromCharCode(_ch);
      if (_isPunctuation(ch)) {
        tokens.add(Token(text: ch, index: charOriginalIndex[count]));
        count += ch.length;
        startNewWord = true;
      } else {
        if (startNewWord) {
          tokens.add(Token(text: '', index: charOriginalIndex[count]));
          startNewWord = false;
        }
        tokens[tokens.length - 1]._text += ch;
        count += ch.length;
      }
    }
    return tokens;
  }

  /**
   * Generate tokens for the given vocalbuary.
   * @param text text to be tokenized.
   */
  List<int> tokenize(String text) {
    // Source:
    // https://github.com/google-research/bert/blob/88a817c37f788702a363ff935fd173b6dc6ac0d6/tokenization.py#L311

    final List<int> outputTokens = [];

    final words = this.processInput(text);
    words.forEach((word) {
      if (word.text != CLS_TOKEN && word.text != SEP_TOKEN) {
        word._text = '${SEPERATOR}${unorm.nfkc(word.text)}';
      }
    });

    for (int i = 0; i < words.length; i++) {
      final chars = <String>[];
      for (final symbol in words[i].text.codeUnits) {
        chars.add(String.fromCharCode(symbol));
      }

      bool isUnknown = false;
      int start = 0;
      final List<int> subTokens = [];

      final charsLength = chars.length;

      while (start < charsLength) {
        int end = charsLength;
        int? currIndex;

        while (start < end) {
          final substr = chars.slice(start, end).join('');

          final match = this.trie.find(substr);
          if (match != null && match.end) {
            currIndex = match.getWord().index;
            break;
          }

          end = end - 1;
        }

        if (currIndex == null) {
          isUnknown = true;
          break;
        }

        subTokens.add(currIndex);
        start = end;
      }

      if (isUnknown) {
        outputTokens.add(UNK_INDEX);
      } else {
        outputTokens.addAll(subTokens);
      }
    }

    return outputTokens;
  }
}

Future<BertTokenizer> loadTokenizer() async {
  final tokenizer = BertTokenizer();
  await tokenizer.load();
  return tokenizer;
}
