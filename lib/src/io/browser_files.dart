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

/**
 * IOHandlers related to files, such as browser-triggered file downloads,
 * user-selected files in browser.
 */

import 'dart:convert';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:universal_html/html.dart' as html;
import 'package:universal_io/io.dart' as io;

import '../environment.dart';
import 'io.dart';

extension FileName on io.File {
  String get name => uri.pathSegments.last;
}

// import '../flags';
// import {env} from '../environment';

// import {basename, concatenateArrayBuffers, getModelArtifactsForJSON, getModelArtifactsInfoForJSON, getModelJSONForModelArtifacts} from './io_utils';
// import {IORouter, IORouterRegistry} from './router_registry';
// import {IOHandler, ModelArtifacts, ModelJSON, SaveResult, WeightsManifestConfig, WeightsManifestEntry} from './types';

const DEFAULT_FILE_NAME_PREFIX = 'model';
const DEFAULT_JSON_EXTENSION_NAME = '.json';
const DEFAULT_WEIGHT_DATA_EXTENSION_NAME = '.weights.bin';

Future<T> _defer<T>(T Function() f) {
  return Future.delayed(Duration.zero, f);
}

class BrowserDownloads implements IOHandler {
  late final String modelJsonFileName;
  late final String weightDataFileName;
  html.AnchorElement? modelJsonAnchor;
  html.AnchorElement? weightDataAnchor;

  static const URL_SCHEME = 'downloads://';

  BrowserDownloads(String fileNamePrefix) {
    if (!env().getBool('IS_BROWSER')) {
      // TODO(cais): Provide info on what IOHandlers are available under the
      //   current environment.
      throw Exception(
          'browserDownloads() cannot proceed because the current environment ' +
              'is not a browser.');
    }

    if (fileNamePrefix.startsWith(BrowserDownloads.URL_SCHEME)) {
      fileNamePrefix =
          fileNamePrefix.substring(BrowserDownloads.URL_SCHEME.length);
    }
    if (fileNamePrefix == null || fileNamePrefix.length == 0) {
      fileNamePrefix = DEFAULT_FILE_NAME_PREFIX;
    }

    this.modelJsonFileName = fileNamePrefix + DEFAULT_JSON_EXTENSION_NAME;
    this.weightDataFileName =
        fileNamePrefix + DEFAULT_WEIGHT_DATA_EXTENSION_NAME;
  }

  final load = null;

  late final save = _save;
  Future<SaveResult> _save(ModelArtifacts modelArtifacts) async {
    if (!kIsWeb) {
      throw Exception('Browser downloads are not supported in ' +
          'this environment since `document` is not present');
    }
    final weightsURL = html.Url.createObjectUrl(
        html.Blob([modelArtifacts.weightData], 'application/octet-stream'));

    if (modelArtifacts.modelTopology is ByteBuffer) {
      throw Exception(
          'BrowserDownloads.save() does not support saving model topology ' +
              'in binary formats yet.');
    } else {
      final weightsManifest = [
        WeightsManifestGroupConfig(
          paths: ['./' + this.weightDataFileName],
          weights: modelArtifacts.weightSpecs!,
        )
      ];
      final modelJSON =
          getModelJSONForModelArtifacts(modelArtifacts, weightsManifest);
      final modelJsonURL = html.Url.createObjectUrl(
          html.Blob([jsonEncode(modelJSON)], 'application/json'));

      // If anchor elements are not provided, create them without attaching them
      // to parents, so that the downloaded file names can be controlled.
      final jsonAnchor = this.modelJsonAnchor ?? html.AnchorElement();
      jsonAnchor.download = this.modelJsonFileName;
      jsonAnchor.href = modelJsonURL;
      // Trigger downloads by evoking a click event on the download anchors.
      // When multiple downloads are started synchronously, Firefox will only
      // save the last one.
      await _defer(() => jsonAnchor.dispatchEvent(html.MouseEvent('click')));

      if (modelArtifacts.weightData != null) {
        final weightDataAnchor = this.weightDataAnchor ?? html.AnchorElement();
        weightDataAnchor.download = this.weightDataFileName;
        weightDataAnchor.href = weightsURL;
        await _defer(
            () => weightDataAnchor.dispatchEvent(html.MouseEvent('click')));
      }

      return SaveResult(
          modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts));
    }
  }
}

class BrowserFiles implements IOHandler {
  late final io.File jsonFile;
  late final List<io.File> weightsFiles;

  BrowserFiles(List<io.File> files) {
    if (files == null || files.length < 1) {
      throw Exception(
          'When calling browserFiles, at least 1 file is required, ' +
              'but received ${files}');
    }
    this.jsonFile = files[0];
    this.weightsFiles = files.slice(1);
  }

  final save = null;

  late final load = _load;
  Future<ModelArtifacts> _load() async {
    final ModelJSON modelJSON;
    try {
      final str = await this.jsonFile.readAsString();
      modelJSON = ModelJSON.fromJson(jsonDecode(str));
    } catch (_) {
      throw Exception("Failed to read model topology and weights manifest JSON " +
          "from file '${this.jsonFile.name}'. BrowserFiles supports loading " +
          "Keras-style tf.Model artifacts only.");
    }

    // tslint:disable-next-line:no-any

    final modelTopology = modelJSON.modelTopology;
    if (modelTopology == null) {
      throw Exception(
          'modelTopology field is missing from file ${this.jsonFile.name}');
    }

    final weightsManifest = modelJSON.weightsManifest;
    if (weightsManifest == null) {
      throw Exception(
          'weightManifest field is missing from file ${this.jsonFile.name}');
    }

    if (this.weightsFiles.length == 0) {
      return ModelArtifacts(modelTopology: modelTopology);
    }

    final modelArtifactsFuture = getModelArtifactsForJSON(
        modelJSON, (weightsManifest) => this._loadWeights(weightsManifest));
    return modelArtifactsFuture;
  }

  Future<EncodedWeights> _loadWeights(WeightsManifestConfig weightsManifest) {
    final weightSpecs = <WeightsManifestEntry>[];
    final paths = <String>[];
    for (final entry in weightsManifest) {
      weightSpecs.addAll(entry.weights);
      paths.addAll(entry.paths);
    }

    final pathToFile = this._checkManifestAndWeightFiles(weightsManifest);

    final promises =
        paths.map((path) => this._loadWeightsFile(path, pathToFile[path]!));

    return Future.wait(promises).then((buffers) => EncodedWeights(
        specs: weightSpecs, data: concatenateArrayBuffers(buffers)));
  }

  Future<ByteBuffer> _loadWeightsFile(String path, io.File file) async {
    final weightData = await file.readAsBytes();
    return weightData.buffer;
    // return Future((resolve, reject) {
    //   final weightFileReader = new FileReader();
    //   weightFileReader.onload = (event: Event) {
    //     // tslint:disable-next-line:no-any
    //     final weightData = (event.target as any).result as ArrayBuffer;
    //     resolve(weightData);
    //   };
    //   weightFileReader.onerror = error =>
    //       reject("Failed to weights data from file of path '${path}'.");
    //   weightFileReader.readAsArrayBuffer(file);
    // });
  }

  /**
   * Check the compatibility between weights manifest and weight files.
   */
  Map<String, io.File> _checkManifestAndWeightFiles(
      WeightsManifestConfig manifest) {
    final basenames = <String>[];
    final fileNames =
        this.weightsFiles.map((file) => basename(file.name)).toList();
    final Map<String, io.File> pathToFile = {};
    for (final group in manifest) {
      group.paths.forEach((path) {
        final pathBasename = basename(path);
        if (basenames.indexOf(pathBasename) != -1) {
          throw Exception(
              "Duplicate file basename found in weights manifest: " +
                  "'${pathBasename}'");
        }
        basenames.add(pathBasename);
        if (fileNames.indexOf(pathBasename) == -1) {
          throw Exception(
              "Weight file with basename '${pathBasename}' is not provided.");
        } else {
          pathToFile[path] = this.weightsFiles[fileNames.indexOf(pathBasename)];
        }
      });
    }

    if (basenames.length != this.weightsFiles.length) {
      throw Exception('Mismatch in the number of files in weights manifest ' +
          '(${basenames.length}) and the number of weight files provided ' +
          '(${this.weightsFiles.length}).');
    }
    return pathToFile;
  }
}

IOHandler? browserDownloadsRouter(List<String> url, LoadOptions? _) {
  if (!env().getBool('IS_BROWSER')) {
    return null;
  } else {
    if (url.length == 1 && url.first.startsWith(BrowserDownloads.URL_SCHEME)) {
      return browserDownloads(
          url.first.substring(BrowserDownloads.URL_SCHEME.length));
    } else {
      return null;
    }
  }
}

/**
 * Creates an IOHandler that triggers file downloads from the browser.
 *
 * The returned `IOHandler` instance can be used as model exporting methods such
 * as `tf.Model.save` and supports only saving.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * const saveResult = await model.save('downloads://mymodel');
 * // This will trigger downloading of two files:
 * //   'mymodel.json' and 'mymodel.weights.bin'.
 * console.log(saveResult);
 * ```
 *
 * @param fileNamePrefix Prefix name of the files to be downloaded. For use with
 *   `tf.Model`, `fileNamePrefix` should follow either of the following two
 *   formats:
 *   1. `null` or `undefined`, in which case the default file
 *      names will be used:
 *      - 'model.json' for the JSON file containing the model topology and
 *        weights manifest.
 *      - 'model.weights.bin' for the binary file containing the binary weight
 *        values.
 *   2. A single string or an Array of a single string, as the file name prefix.
 *      For example, if `'foo'` is provided, the downloaded JSON
 *      file and binary weights file will be named 'foo.json' and
 *      'foo.weights.bin', respectively.
 * @param config Additional configuration for triggering downloads.
 * @returns An instance of `BrowserDownloads` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
IOHandler browserDownloads([String fileNamePrefix = 'model']) {
  return BrowserDownloads(fileNamePrefix);
}

/**
 * Creates an IOHandler that loads model artifacts from user-selected files.
 *
 * This method can be used for loading from files such as user-selected files
 * in the browser.
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * // Note: This code snippet won't run properly without the actual file input
 * //   elements in the HTML DOM.
 *
 * // Suppose there are two HTML file input (`<input type="file" ...>`)
 * // elements.
 * const uploadJSONInput = document.getElementById('upload-json');
 * const uploadWeightsInput = document.getElementById('upload-weights');
 * const model = await tf.loadLayersModel(tf.io.browserFiles(
 *     [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
 * ```
 *
 * @param files `File`s to load from. Currently, this function supports only
 *   loading from files that contain Keras-style models (i.e., `tf.Model`s), for
 *   which an `Array` of `File`s is expected (in that order):
 *   - A JSON file containing the model topology and weight manifest.
 *   - Optionally, One or more binary files containing the binary weights.
 *     These files must have names that match the paths in the `weightsManifest`
 *     contained by the aforementioned JSON file, or errors will be thrown
 *     during loading. These weights files have the same format as the ones
 *     generated by `tensorflowjs_converter` that comes with the `tensorflowjs`
 *     Python PIP package. If no weights files are provided, only the model
 *     topology will be loaded from the JSON file above.
 * @returns An instance of `Files` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
IOHandler browserFiles(List<io.File> files) {
  return BrowserFiles(files);
}
