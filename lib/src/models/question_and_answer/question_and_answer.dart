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
import 'dart:math' as Math;
import 'package:collection/collection.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:tensorflow_wasm/converter.dart' as tfconv;

// import {BertTokenizer, CLS_INDEX, loadTokenizer, SEP_INDEX, Token} from './bert_tokenizer';
import 'bert_tokenizer.dart';

const MODEL_URL = 'https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1';
const INPUT_SIZE = 384;
const MAX_ANSWER_LEN = 32;
const MAX_QUERY_LEN = 64;
const MAX_SEQ_LEN = 384;
const PREDICT_ANSWER_NUM = 5;
const OUTPUT_OFFSET = 1;
// This is threshold value for determining if a question is irrelevant to the
// context. This value comes from the QnA model, and is generated by the
// training process based on the SQUaD 2.0 dataset.
const NO_ANSWER_THRESHOLD = 4.3980759382247925;

abstract class QuestionAndAnswer {
  /**
   * Given the question and context, find the best answers.
   * @param question the question to find answers for.
   * @param context context where the answers are looked up from.
   * @return array of answers
   */
  Future<List<Answer>> findAnswers(String question, String context);
}

/**
 * MobileBert model loading is configurable using the following config
 * dictionary.
 *
 * `modelUrl`: An optional string that specifies custom url of the model. This
 * is useful for area/countries that don't have access to the model hosted on
 * GCP.
 */
class ModelConfig {
  /**
   * An optional string that specifies custom url of the model. This
   * is useful for area/countries that don't have access to the model hosted on
   * GCP.
   */
  final String modelUrl;
  /**
   * Wheter the url is from tfhub.
   */
  final bool fromTFHub;

  const ModelConfig({
    required this.modelUrl,
    required this.fromTFHub,
  });
}

/**
 * Answer object returned by the model.
 * `text`: string, the text of the answer.
 * `startIndex`: number, the index of the starting character of the answer in
 *     the passage.
 * `endIndex`: number, index of the last character of the answer text.
 * `score`: number, indicates the confident
 * level.
 */
class Answer {
  final String text;
  final int startIndex;
  final int endIndex;
  final double score;

  Answer({
    required this.text,
    required this.startIndex,
    required this.endIndex,
    required this.score,
  });
}

class Feature {
  final List<int> inputIds;
  final List<int> inputMask;
  final List<int> segmentIds;
  final List<Token> origTokens;
  final Map<int, int> tokenToOrigMap;

  Feature({
    required this.inputIds,
    required this.inputMask,
    required this.segmentIds,
    required this.origTokens,
    required this.tokenToOrigMap,
  });
}

class AnswerIndex {
  final int start;
  final int end;
  final double score;

  AnswerIndex({
    required this.start,
    required this.end,
    required this.score,
  });
}

class _Span {
  final int start;
  final int length;
  _Span({
    required this.start,
    required this.length,
  });
}

class QuestionAndAnswerImpl implements QuestionAndAnswer {
  late final tfconv.GraphModel model;
  late final BertTokenizer tokenizer;
  final ModelConfig modelConfig;

  QuestionAndAnswerImpl([
    ModelConfig? modelConfig,
  ]) : modelConfig = modelConfig ??
            const ModelConfig(modelUrl: MODEL_URL, fromTFHub: true);

  List<Feature> _process(
      String query, String context, int maxQueryLen, int maxSeqLen,
      [int docStride = 128]) {
    // always add the question mark to the end of the query.
    query = query.replaceAll(RegExp(r'\?'), '');
    query = query.trim();
    query = query + '?';

    final queryTokens = this.tokenizer.tokenize(query);
    if (queryTokens.length > maxQueryLen) {
      throw Exception(
          'The length of question token exceeds the limit (${maxQueryLen}).');
    }

    final origTokens = this.tokenizer.processInput(context.trim());
    final List<int> tokenToOrigIndex = [];
    final List<int> allDocTokens = [];
    for (int i = 0; i < origTokens.length; i++) {
      final token = origTokens[i].text;
      final subTokens = this.tokenizer.tokenize(token);
      for (int j = 0; j < subTokens.length; j++) {
        final subToken = subTokens[j];
        tokenToOrigIndex.add(i);
        allDocTokens.add(subToken);
      }
    }
    // The -3 accounts for [CLS], [SEP] and [SEP]
    final maxContextLen = maxSeqLen - queryTokens.length - 3;

    // We can have documents that are longer than the maximum sequence
    // length. To deal with this we do a sliding window approach, where we
    // take chunks of the up to our max length with a stride of
    // `doc_stride`.
    final List<_Span> docSpans = [];
    int startOffset = 0;
    while (startOffset < allDocTokens.length) {
      int length = allDocTokens.length - startOffset;
      if (length > maxContextLen) {
        length = maxContextLen;
      }
      docSpans.add(_Span(start: startOffset, length: length));
      if (startOffset + length == allDocTokens.length) {
        break;
      }
      startOffset += Math.min(length, docStride);
    }

    final features = docSpans.map((docSpan) {
      final List<int> tokens = [];
      final List<int> segmentIds = [];
      final Map<int, int> tokenToOrigMap = {};
      tokens.add(CLS_INDEX);
      segmentIds.add(0);
      for (int i = 0; i < queryTokens.length; i++) {
        final queryToken = queryTokens[i];
        tokens.add(queryToken);
        segmentIds.add(0);
      }
      tokens.add(SEP_INDEX);
      segmentIds.add(0);
      for (int i = 0; i < docSpan.length; i++) {
        final splitTokenIndex = i + docSpan.start;
        final docToken = allDocTokens[splitTokenIndex];
        tokens.add(docToken);
        segmentIds.add(1);
        tokenToOrigMap[tokens.length] = tokenToOrigIndex[splitTokenIndex];
      }
      tokens.add(SEP_INDEX);
      segmentIds.add(1);
      final inputIds = tokens;
      final inputMask = inputIds.map((id) => 1).toList();
      while ((inputIds.length < maxSeqLen)) {
        inputIds.add(0);
        inputMask.add(0);
        segmentIds.add(0);
      }
      return Feature(
        inputIds: inputIds,
        inputMask: inputMask,
        segmentIds: segmentIds,
        origTokens: origTokens,
        tokenToOrigMap: tokenToOrigMap,
      );
    }).toList();
    return features;
  }

  Future<void> load() async {
    this.model = await tfconv.loadGraphModel(
      tfconv.ModelHandler.fromUrl(this.modelConfig.modelUrl),
      tfconv.LoadOptions(fromTFHub: this.modelConfig.fromTFHub),
    );
    // warm up the backend
    final batchSize = 1;
    final inputIds = tf.ones([batchSize, INPUT_SIZE], 'int32');
    final segmentIds = tf.ones([1, INPUT_SIZE], 'int32');
    final inputMask = tf.ones([1, INPUT_SIZE], 'int32');
    this.model.execute(tf.TensorMap({
          'input_ids': inputIds,
          'segment_ids': segmentIds,
          'input_mask': inputMask,
          'global_step': tf.scalar(1, 'int32')
        }));

    this.tokenizer = await loadTokenizer();
  }

  /**
   * Given the question and context, find the best answers.
   * @param question the question to find answers for.
   * @param context context where the answers are looked up from.
   * @return array of answers
   */
  @override
  Future<List<Answer>> findAnswers(String question, String context) async {
    // if (question == null || context == null) {
    //   throw Exception('The input to findAnswers call is null, ' +
    //       'please pass a string as input.');
    // }

    final features =
        this._process(question, context, MAX_QUERY_LEN, MAX_SEQ_LEN);
    final inputIdArray = features.map((f) => f.inputIds).toList();
    final segmentIdArray = features.map((f) => f.segmentIds).toList();
    final inputMaskArray = features.map((f) => f.inputMask).toList();
    final globalStep = tf.scalar(1, 'int32');
    final batchSize = features.length;
    final result = tf.tidy(() {
      final inputIds =
          tf.tensor2d(inputIdArray, [batchSize, INPUT_SIZE], 'int32');
      final segmentIds =
          tf.tensor2d(segmentIdArray, [batchSize, INPUT_SIZE], 'int32');
      final inputMask =
          tf.tensor2d(inputMaskArray, [batchSize, INPUT_SIZE], 'int32');
      return this.model.execute(
          tf.TensorMap({
            'input_ids': inputIds,
            'segment_ids': segmentIds,
            'input_mask': inputMask,
            'global_step': globalStep
          }),
          ['start_logits', 'end_logits']) as List<tf.Tensor2D>;
    });
    final logits = (await Future.wait([
      result[0].array(),
      result[1].array(),
    ]))
        .cast<List>();
    // dispose all intermediate tensors
    globalStep.dispose();
    result[0].dispose();
    result[1].dispose();

    final List<Answer> answers = [];
    for (int i = 0; i < batchSize; i++) {
      answers.addAll(this.getBestAnswers(
        (logits[0][i] as List).cast(),
        (logits[1][i] as List).cast(),
        features[i].origTokens,
        features[i].tokenToOrigMap,
        context,
        i,
      ));
    }

    answers.sort((logitA, logitB) => (logitB.score - logitA.score).round());

    return answers.length > PREDICT_ANSWER_NUM
        ? answers.slice(0, PREDICT_ANSWER_NUM)
        : answers;
  }

  /**
   * Find the Best N answers & logits from the logits array and input feature.
   * @param startLogits start index for the answers
   * @param endLogits end index for the answers
   * @param origTokens original tokens of the passage
   * @param tokenToOrigMap token to index mapping
   */
  List<Answer> getBestAnswers(
    List<double> startLogits,
    List<double> endLogits,
    List<Token> origTokens,
    Map<int, int> tokenToOrigMap,
    String context, [
    int docIndex = 0,
  ]) {
    // Model uses the closed interval [start, end] for indices.
    final startIndexes = this.getBestIndex(startLogits);
    final endIndexes = this.getBestIndex(endLogits);
    final List<AnswerIndex> origResults = [];
    startIndexes.forEach((start) {
      endIndexes.forEach((end) {
        if (!const [null, 0].contains(tokenToOrigMap[start + OUTPUT_OFFSET]) &&
            !const [null, 0].contains(tokenToOrigMap[end + OUTPUT_OFFSET]) &&
            end >= start) {
          final length = end - start + 1;
          if (length < MAX_ANSWER_LEN) {
            origResults.add(AnswerIndex(
              start: start,
              end: end,
              score: startLogits[start] + endLogits[end],
            ));
          }
        }
      });
    });

    origResults.sort((a, b) => (b.score - a.score).round());
    final List<Answer> answers = [];
    for (int i = 0; i < origResults.length; i++) {
      if (i >= PREDICT_ANSWER_NUM ||
          origResults[i].score < NO_ANSWER_THRESHOLD) {
        break;
      }

      String convertedText = '';
      int startIndex = 0;
      int endIndex = 0;
      if (origResults[i].start > 0) {
        final _c = this.convertBack(origTokens, tokenToOrigMap,
            origResults[i].start, origResults[i].end, context);

        convertedText = _c.convertedText;
        startIndex = _c.startIndex;
        endIndex = _c.endIndex;
      } else {
        convertedText = '';
      }
      answers.add(Answer(
          text: convertedText,
          score: origResults[i].score,
          startIndex: startIndex,
          endIndex: endIndex));
    }
    return answers;
  }

  /** Get the n-best logits from a list of all the logits. */
  List<int> getBestIndex(List<double> logits) {
    final tmpList = <List<num>>[];
    for (int i = 0; i < MAX_SEQ_LEN; i++) {
      tmpList.add([i, i, logits[i]]);
    }
    tmpList.sort((a, b) => (b[2] - a[2]).round());

    final List<int> indexes = [];
    for (int i = 0; i < PREDICT_ANSWER_NUM; i++) {
      indexes.add(tmpList[i][0] as int);
    }

    return indexes;
  }

  /** Convert the answer back to original text form. */
  _ConvertedIndex convertBack(
    List<Token> origTokens,
    Map<int, int> tokenToOrigMap,
    int start,
    int end,
    String context,
  ) {
    // Shifted index is: index of logits + offset.
    final shiftedStart = start + OUTPUT_OFFSET;
    final shiftedEnd = end + OUTPUT_OFFSET;
    final startIndex = tokenToOrigMap[shiftedStart]!;
    final endIndex = tokenToOrigMap[shiftedEnd]!;
    final startCharIndex = origTokens[startIndex].index;

    final endCharIndex = endIndex < origTokens.length - 1
        ? origTokens[endIndex + 1].index - 1
        : origTokens[endIndex].index + origTokens[endIndex].text.length;

    return _ConvertedIndex(
      convertedText: context.substring(startCharIndex, endCharIndex + 1).trim(),
      startIndex: startCharIndex,
      endIndex: endCharIndex,
    );
  }
}

Future<QuestionAndAnswer> load([ModelConfig? modelConfig]) async {
  final mobileBert = QuestionAndAnswerImpl(modelConfig);
  await mobileBert.load();
  return mobileBert;
}

class _ConvertedIndex {
  final String convertedText;
  final int startIndex;
  final int endIndex;

  _ConvertedIndex({
    required this.convertedText,
    required this.startIndex,
    required this.endIndex,
  });
}
