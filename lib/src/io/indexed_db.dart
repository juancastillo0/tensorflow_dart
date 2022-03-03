/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// import '../flags';

// import {env} from '../environment';

// import {getModelArtifactsInfoForJSON} from './io_utils';
// import {IORouter, IORouterRegistry} from './router_registry';
// import {IOHandler, ModelArtifacts, ModelArtifactsInfo, ModelStoreManager, SaveResult} from './types';

import 'dart:async';
import 'dart:typed_data';

import 'io.dart';

import '../environment.dart';
import 'package:universal_html/indexed_db.dart' as idb;
import 'package:universal_html/html.dart' as html;

const DATABASE_NAME = 'tensorflowjs';
const DATABASE_VERSION = 1;

// Model data and ModelArtifactsInfo (metadata) are stored in two separate
// stores for efficient access of the list of stored models and their metadata.
// 1. The object store for model data: topology, weights and weight manifests.
const MODEL_STORE_NAME = 'models_store';
// 2. The object store for ModelArtifactsInfo, including meta-information such
//    as the type of topology (JSON vs binary), byte size of the topology, byte
//    size of the weights, etc.
const INFO_STORE_NAME = 'model_info_store';

/**
 * Delete the entire database for tensorflow.js, including the models store.
 */
Future<void> deleteDatabase() async {
  final idbFactory = _getIndexedDBFactory();

  // return new Future<void>((resolve, reject) => {
  await idbFactory.deleteDatabase(DATABASE_NAME);
  // deleteRequest.onsuccess = () => resolve();
  // deleteRequest.onerror = error => reject(error);
  // });
}

idb.IdbFactory _getIndexedDBFactory() {
  if (!env().getBool('IS_BROWSER')) {
    // TODO(cais): Add more info about what IOHandler subtypes are available.
    //   Maybe point to a doc page on the web and/or automatically determine
    //   the available IOHandlers and print them in the error message.
    throw Exception(
        'Failed to obtain IndexedDB factory because the current environment' +
            'is not a web browser.');
  }
  // tslint:disable-next-line:no-any
  // final theWindow: any = typeof window === 'undefined' ? self : window;
  // final factory = theWindow.indexedDB || theWindow.mozIndexedDB ||
  //     theWindow.webkitIndexedDB || theWindow.msIndexedDB ||
  //     theWindow.shimIndexedDB;
  if (html.window.indexedDB == null || !idb.IdbFactory.supported) {
    throw Exception(
        'The current browser does not appear to support IndexedDB.');
  }
  return html.window.indexedDB!;
}

void _setUpDatabase(idb.Request openRequest) {
  final db = openRequest.result as idb.Database;
  db.createObjectStore(MODEL_STORE_NAME, keyPath: 'modelPath');
  db.createObjectStore(INFO_STORE_NAME, keyPath: 'modelPath');
}

/**
 * IOHandler subclass: Browser IndexedDB.
 *
 * See the doc string of `browserIndexedDB` for more details.
 */
class BrowserIndexedDB implements IOHandler {
  final idb.IdbFactory indexedDB;
  final String modelPath;

  static const URL_SCHEME = 'indexeddb://';

  BrowserIndexedDB(this.modelPath) : indexedDB = _getIndexedDBFactory() {
    if (modelPath == null || modelPath.isEmpty) {
      throw Exception(
          'For IndexedDB, modelPath must not be null, undefined or empty.');
    }
  }
  late final save = _save;
  Future<SaveResult> _save(ModelArtifacts modelArtifacts) async {
    // TODO(cais): Support saving GraphDef models.
    if (modelArtifacts.modelTopology is ByteBuffer) {
      throw Exception(
          'BrowserLocalStorage.save() does not support saving model topology ' +
              'in binary formats yet.');
    }

    return this._databaseAction(this.modelPath, modelArtifacts)
        as Future<SaveResult>;
  }

  late final load = _load;
  Future<ModelArtifacts> _load() async {
    return this._databaseAction(this.modelPath) as Future<ModelArtifacts>;
  }

  /**
   * Perform database action to put model artifacts into or read model artifacts
   * from IndexedDB object store.
   *
   * Whether the action is put or get depends on whether `modelArtifacts` is
   * specified. If it is specified, the action will be put; otherwise the action
   * will be get.
   *
   * @param modelPath A unique string path for the model.
   * @param modelArtifacts If specified, it will be the model artifacts to be
   *   stored in IndexedDB.
   * @returns A `Future` of `SaveResult`, if the action is put, or a `Future`
   *   of `ModelArtifacts`, if the action is get.
   */
  Future<Object>
      // ModelArtifacts|SaveResult
      _databaseAction(String modelPath,
          [ModelArtifacts? modelArtifacts]) async {
    final db = await this.indexedDB.open(
          DATABASE_NAME,
          version: DATABASE_VERSION,
          onUpgradeNeeded: (e) => _setUpDatabase(e.target),
        );

    if (modelArtifacts == null) {
      // Read model out from object store.
      final modelTx = db.transaction(MODEL_STORE_NAME, 'readonly');
      modelTx.onComplete.first.then((value) => db.close());
      final modelStore = modelTx.objectStore(MODEL_STORE_NAME);
      try {
        final result = await modelStore.getObject(this.modelPath);
        if (result == null) {
          throw Exception("Cannot find model with path '${this.modelPath}' " +
              "in IndexedDB.");
        }
        return result['modelArtifacts'];
      } catch (e) {
        db.close();
        rethrow;
      }
    } else {
      // Put model into object store.
      final ModelArtifactsInfo modelArtifactsInfo =
          getModelArtifactsInfoForJSON(modelArtifacts);
      // First, put ModelArtifactsInfo into info store.
      final infoTx = db.transaction(INFO_STORE_NAME, 'readwrite');
      idb.Transaction? modelTx;
      infoTx.onComplete.first.then((value) {
        if (modelTx == null) {
          db.close();
        } else {
          modelTx.onComplete.first.then((_) => db.close());
        }
      });
      var infoStore = infoTx.objectStore(INFO_STORE_NAME);
      try {
        final putInfoRequest = await infoStore.put({
          'modelPath': this.modelPath,
          'modelArtifactsInfo': modelArtifactsInfo
        });

        // Second, put model data into model store.
        modelTx = db.transaction(MODEL_STORE_NAME, 'readwrite');
        final modelStore = modelTx.objectStore(MODEL_STORE_NAME);
        try {
          final putModelRequest = await modelStore.put({
            'modelPath': this.modelPath,
            'modelArtifacts': modelArtifacts,
            'modelArtifactsInfo': modelArtifactsInfo
          });
          return {'modelArtifactsInfo': modelArtifactsInfo};
        } catch (error) {
          // If the put-model request fails, roll back the info entry as
          // well.
          infoStore = infoTx.objectStore(INFO_STORE_NAME);
          try {
            await infoStore.delete(this.modelPath);
          } catch (_) {}
          db.close();
          rethrow;
        }
      } catch (e) {
        db.close();
        rethrow;
      }
    }
  }
}

IOHandler? indexedDBRouter(List<String> url, LoadOptions? _) {
  if (!env().getBool('IS_BROWSER')) {
    return null;
  } else {
    if (url.length == 1 && url.first.startsWith(BrowserIndexedDB.URL_SCHEME)) {
      return browserIndexedDB(
          url.first.substring(BrowserIndexedDB.URL_SCHEME.length));
    } else {
      return null;
    }
  }
}

/**
 * Creates a browser IndexedDB IOHandler for saving and loading models.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save('indexeddb://MyModel'));
 * console.log(saveResult);
 * ```
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `BrowserIndexedDB` (sublcass of `IOHandler`),
 *   which can be used with, e.g., `tf.Model.save`.
 */
IOHandler browserIndexedDB(String modelPath) {
  return BrowserIndexedDB(modelPath);
}

String _maybeStripScheme(String key) {
  return key.startsWith(BrowserIndexedDB.URL_SCHEME)
      ? key.substring(BrowserIndexedDB.URL_SCHEME.length)
      : key;
}

class BrowserIndexedDBManager implements ModelStoreManager {
  final idb.IdbFactory indexedDB;

  BrowserIndexedDBManager() : this.indexedDB = _getIndexedDBFactory();

  Future<Map<String, ModelArtifactsInfo>> listModels() async {
    final db = await this.indexedDB.open(
          DATABASE_NAME,
          version: DATABASE_VERSION,
          onUpgradeNeeded: (e) => _setUpDatabase(e.target),
        );

    final tx = db.transaction(INFO_STORE_NAME, 'readonly');
    tx.onComplete.first.then((_) => db.close());
    final store = tx.objectStore(INFO_STORE_NAME);
    // tslint:disable:max-line-length
    // Need to cast `store` as `any` here because TypeScript's DOM
    // library does not have the `getAll()` method even though the
    // method is supported in the latest version of most mainstream
    // browsers:
    // https://developer.mozilla.org/en-US/docs/Web/API/IDBObjectStore/getAll
    // tslint:enable:max-line-length
    // tslint:disable-next-line:no-any
    final getAllInfoRequest = store.getAll(null);
    final comp = Completer<Map<String, ModelArtifactsInfo>>();
    getAllInfoRequest.onSuccess.first.then((e) {
      final Map<String, ModelArtifactsInfo> out = {};
      for (final item in getAllInfoRequest.result) {
        out[item['modelPath']] = item['modelArtifactsInfo'];
      }
      comp.complete(out);
    });
    getAllInfoRequest.onError.first.then((error) {
      db.close();
      return comp.completeError(getAllInfoRequest.error ?? error);
    });
    return comp.future;
  }

  Future<ModelArtifactsInfo> removeModel(String path) async {
    path = _maybeStripScheme(path);

    final db = await this.indexedDB.open(
          DATABASE_NAME,
          version: DATABASE_VERSION,
          onUpgradeNeeded: (e) => _setUpDatabase(e.target),
        );

    final infoTx = db.transaction(INFO_STORE_NAME, 'readwrite');

    final completer = Completer<ModelArtifactsInfo>();

    void reject(Object error) {
      completer.completeError(error);
    }

    idb.Transaction? modelTx;
    infoTx.onComplete.first.then((value) {
      if (modelTx == null) {
        db.close();
      } else {
        modelTx!.onComplete.first.then((_) => db.close());
      }
    });
    final infoStore = infoTx.objectStore(INFO_STORE_NAME);

    final getInfoRequest = infoStore.getKey(path);

    getInfoRequest.onSuccess.first.then((_) async {
      if (getInfoRequest.result == null) {
        db.close();
        return reject(Exception(
            "Cannot find model with path '${path}' " + "in IndexedDB."));
      } else {
        void deleteModelData() {
          // Second, delete the entry in the model store.
          modelTx = db.transaction(MODEL_STORE_NAME, 'readwrite');
          final modelStore = modelTx!.objectStore(MODEL_STORE_NAME);
          modelStore
              .delete(path)
              .then((value) =>
                  completer.complete(getInfoRequest.result.modelArtifactsInfo))
              .onError((error, stackTrace) => reject(error!));
        }

        try {
          // First, delete the entry in the info store.
          await infoStore.delete(path);
          deleteModelData();
        } catch (error) {
          // Proceed with deleting model data regardless of whether deletion
          // of info data succeeds or not.
          deleteModelData();
          db.close();
          return reject(error);
        }
      }
    });
    getInfoRequest.onError.first.then((error) {
      db.close();
      return reject(error);
    });

    return completer.future;
  }
}
