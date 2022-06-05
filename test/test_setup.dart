import 'package:tensorflow_wasm/backend_wasm.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart' as tf;
import 'package:logging/logging.dart';

Future<void> setUpWasmBackend() async {
  Logger.root.level = Level.CONFIG; // defaults to Level.INFO
  Logger.root.onRecord.listen((record) {
    print('${record.level.name}: ${record.time}: ${record.message}');
  });

  setWasmPaths('./web_examples/web/tensorflow_wasm/');
  await tf.setBackend(wasmBackendFactory);
}
