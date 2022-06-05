import 'package:tensorflow_wasm/backend_wasm.dart';
import 'package:tensorflow_wasm/src/models/question_and_answer/bert_tokenizer.dart';
import 'package:test/test.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

import 'package:logging/logging.dart';

void main() async {
  group('bertTokenizer', () {
    setUpAll(() async {
      Logger.root.level = Level.CONFIG; // defaults to Level.INFO
      Logger.root.onRecord.listen((record) {
        print('${record.level.name}: ${record.time}: ${record.message}');
      });

      setWasmPaths('./web_examples/web/tensorflow_wasm/');
      await tf.setBackend(wasmBackendFactory);
    });

    test('should load', () async {
      final tokenizer = await loadTokenizer();

      expect(tokenizer, isNotNull);
    });

    test('should tokenize', () async {
      final tokenizer = await loadTokenizer();
      final result = tokenizer.tokenize('a new test');

      expect(result, [1037, 2047, 3231]);
    });

    test('should tokenize punctuation', () async {
      final tokenizer = await loadTokenizer();
      final result = tokenizer.tokenize('a new [test]');

      expect(result, [1037, 2047, 1031, 3231, 1033]);
    });

    test('should tokenize empty string', () async {
      final tokenizer = await loadTokenizer();
      final result = tokenizer.tokenize('');

      expect(result, []);
    });

    test('should tokenize control characters', () async {
      final tokenizer = await loadTokenizer();
      final result = tokenizer.tokenize('a new\b\v [test]');
      expect(result, [1037, 100, 1031, 3231, 1033]);
    });

    test('should processInput', () async {
      final tokenizer = await loadTokenizer();
      final result = tokenizer.processInput(' a new\t\v  [test]');
      expect(result, [
        Token(text: 'a', index: 1),
        Token(text: 'new', index: 3),
        Token(text: '[', index: 10),
        Token(text: 'test', index: 11),
        Token(text: ']', index: 15)
      ]);
    });
  });
}
