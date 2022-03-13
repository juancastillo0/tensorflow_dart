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

// import {stringToChars} from '../util';

import '../util.dart';

// [token, score, index]
class OutputNode {
  final List<String> token;
  final int score;
  final int index;

  OutputNode({
    required this.token,
    required this.score,
    required this.index,
  });
}

class TrieNode {
  TrieNode? parent;
  bool end = false;
  Map<String, TrieNode> children = {};
  OutputNode word = OutputNode(index: 0, score: 0, token: []);

  TrieNode();
}

class Trie {
  TrieNode root = TrieNode();

  Trie() {
    this.root = TrieNode();
  }

  /**
   * Inserts a token into the trie.
   */
  insert(String word, int score, int index) {
    var node = this.root;

    final symbols = stringToChars(word);

    for (int i = 0; i < symbols.length; i++) {
      if (node.children.containsKey(symbols[i])) {
        final t = TrieNode();
        node.children[symbols[i]] = t;
        t.parent = node;
        t.word[0] = node.word[0].concat(symbols[i]);
      }

      node = node.children[symbols[i]];
      if (i == symbols.length - 1) {
        node.end = true;
        node.word[1] = score;
        node.word[2] = index;
      }
    }
  }

  /**
   * Returns an array of all tokens starting with ss.
   *
   * @param ss The prefix to match on.
   */
  List<OutputNode> commonPrefixSearch(List<String> ss) {
    final List<OutputNode> output = [];
    var node = this.root.children[ss[0]];

    for (int i = 0; i < ss.length && node != null; i++) {
      if (node.end) {
        output.add(node.word);
      }
      node = node.children[ss[i + 1]];
    }

    if (output.isEmpty) {
      output.add(OutputNode(
        token: [ss[0]],
        score: 0,
        index: 0,
      ));
    }

    return output;
  }
}
