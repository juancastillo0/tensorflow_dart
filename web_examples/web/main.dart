import 'package:mobx/mobx.dart';

import 'package:tensorflow_wasm/backend_wasm.dart' as tfjsWasm;
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:logging/logging.dart';
import 'question_and_answer/main.dart' as question_and_answer;
import 'face_landmarks/main.dart' as face_landmarks;
import 'hand_pose/live_video/main.dart' as hand_pose;

enum ExampleDemo {
  faceLandmark,
  questionAndAnswer,
  handPose,
}

void main() async {
  mainContext.config = ReactiveConfig(
    writePolicy: ReactiveWritePolicy.never,
  );
  Logger.root.level = Level.CONFIG; // defaults to Level.INFO
  Logger.root.onRecord.listen((record) {
    print('${record.level.name}: ${record.time}: ${record.message}');
  });

  tfjsWasm.setWasmPaths(
      'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.versionWasm}/dist/');
  // tfjsWasm.setWasmPaths('/tensorflow_wasm/');

  final setted = await tf.setBackend(tfjsWasm.wasmBackendFactory);
  print('setted wasm $setted');

  tf.tidy(
    () {
      final dd = tf.tensor([1, 2, 3, 0]);
      tf.print_(dd);
      tf.print_(tf.sub(dd, tf.tensor([-1, 2, 3, 0])));
    },
  );

  const demo = ExampleDemo.questionAndAnswer;
  switch (demo) {
    case ExampleDemo.faceLandmark:
      face_landmarks.main();
      break;
    case ExampleDemo.handPose:
      hand_pose.app();
      break;
    case ExampleDemo.questionAndAnswer:
      question_and_answer.main();
      break;
  }
}
