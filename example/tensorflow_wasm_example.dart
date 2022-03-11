import 'dart:io';

import 'package:path/path.dart';
import 'package:tensorflow_wasm/backend_wasm.dart';
import 'package:tensorflow_wasm/converter.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;

void main() async {
  final modelUrl = joinAll([
    '/',
    ...Platform.script.pathSegments.sublistRelaxed(0, -1),
    'model',
    'model.json'
  ]);

  setWasmPaths(
    joinAll([
      '/',
      ...Platform.script.pathSegments.sublistRelaxed(0, -2),
      'bin/',
    ]),
  );
  await tf.setBackend(wasmBackendFactory);
  final model = await loadGraphModel(ModelHandler.fromUrl('file://$modelUrl'));

  print(model.inputs);
  print(model.metadata);
  print(model.outputs);

  final result = await model.executeAsync(tf.TensorList([
    tf.tensor([
      [0.1, 0.2, 0.3, 0.4, 0]
    ]),
    tf.ones([1, 28, 28, 1]),
    tf.scalar(false),
  ]));

  print(result.toTensorList());
  print(result.toTensorList().first.dataSync());
}
