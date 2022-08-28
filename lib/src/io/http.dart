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

import 'dart:convert';
import 'dart:typed_data';

import 'package:tensorflow_wasm/src/environment.dart';
import 'package:tensorflow_wasm/src/io/io_utils.dart';
import 'package:tensorflow_wasm/src/io/router_registry.dart';
import 'package:tensorflow_wasm/src/io/types.dart';
import 'package:tensorflow_wasm/src/io/weights_loader.dart';
import 'package:tensorflow_wasm/src/tensor_util.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:meta/meta.dart';

/**
 * IOHandler implementations based on HTTP requests in the web browser.
 *
 * Uses [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 */

// import {env} from '../environment';

// import {assert} from '../util';
// import {concatenateArrayBuffers, getModelArtifactsForJSON, getModelArtifactsInfoForJSON, getModelJSONForModelArtifacts} from './io_utils';
// import {IORouter, IORouterRegistry} from './router_registry';
// import {IOHandler, LoadOptions, ModelArtifacts, ModelJSON, OnProgressCallback, SaveResult, WeightsManifestConfig, WeightsManifestEntry} from './types';
// import {loadWeightsAsArrayBuffer} from './weights_loader';

final OCTET_STREAM_MIME_TYPE = MediaType('application', 'octet-stream');
final JSON_TYPE = MediaType('application', 'json');

class HTTPRequest implements IOHandler {
  @protected
  final String path;
  @protected
  final http.BaseRequest requestInit;

  late final FetchFn fetch; // private
  final Future<String> Function(String weightName)?
      weightUrlConverter; // private

  final String? weightPathPrefix; // private
  final OnProgressCallback? onProgress; // private

  static const DEFAULT_METHOD = 'GET';
  static final URL_SCHEME_REGEX = RegExp(r'^https?:\/\/');

  HTTPRequest(this.path, [LoadOptions? loadOptions])
      : weightPathPrefix = loadOptions?.weightPathPrefix,
        onProgress = loadOptions?.onProgress,
        weightUrlConverter = loadOptions?.weightUrlConverter,
        requestInit = loadOptions?.requestInit ??
            http.Request(DEFAULT_METHOD, Uri.parse(path)) {
    loadOptions ??= LoadOptions();

    if (loadOptions.fetchFunc != null) {
      assert(
          loadOptions.fetchFunc is Function,
          () =>
              'Must pass a function that matches the signature of ' +
              '`fetch` (see ' +
              'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)');
      this.fetch = loadOptions.fetchFunc!;
    } else {
      this.fetch = env().platform!.fetch;
    }

    assert(path != null && path.length > 0,
        () => 'URL path for http must not be null, undefined or ' + 'empty.');

    if (path is List) {
      assert(
          path.length == 2,
          () =>
              'URL paths for http must have a length of 2, ' +
              '(actual length is ${path.length}).');
    }
    // this.path = path;

    if (loadOptions.requestInit is http.Request &&
        (loadOptions.requestInit as http.Request).contentLength != 0) {
      throw Exception(
          'requestInit is expected to have no pre-existing body, but has one.');
    }
  }

  late final save = _save;
  Future<SaveResult> _save(ModelArtifacts modelArtifacts) async {
    if (modelArtifacts.modelTopology is ByteBuffer) {
      throw Exception(
          'BrowserHTTPRequest.save() does not support saving model topology ' +
              'in binary formats yet.');
    }

    // TODO:  final init = Object.assign({method: this.DEFAULT_METHOD}, this.requestInit);
    final init = http.MultipartRequest(
      this.requestInit.method,
      Uri.parse(this.path),
    );
    if (this.requestInit != null) {
      init.followRedirects = this.requestInit.followRedirects;
      init.maxRedirects = this.requestInit.maxRedirects;
      init.persistentConnection = this.requestInit.persistentConnection;
      init.headers.addAll(this.requestInit.headers);
    }

    final WeightsManifestConfig weightsManifest = [
      WeightsManifestGroupConfig(
        paths: ['./model.weights.bin'],
        weights: modelArtifacts.weightSpecs!,
      )
    ];
    final ModelJSON modelTopologyAndWeightManifest =
        getModelJSONForModelArtifacts(modelArtifacts, weightsManifest);

    init.files.add(http.MultipartFile.fromString(
      'model.json',
      jsonEncode(modelTopologyAndWeightManifest),
      filename: 'model.json',
      contentType: JSON_TYPE,
    ));

    if (modelArtifacts.weightData != null) {
      init.files.add(http.MultipartFile.fromBytes(
        'model.weights.bin',
        modelArtifacts.weightData!.asUint8List(),
        filename: 'model.weights.bin',
        contentType: OCTET_STREAM_MIME_TYPE,
      ));
    }

    final response = await this.fetch(Uri.parse(this.path), init);

    if (response.statusCode < 300) {
      return SaveResult(
        modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts),
        responses: [response],
      );
    } else {
      throw Exception(
          'BrowserHTTPRequest.save() failed due to HTTP response status ' +
              '${response.statusCode}.');
    }
  }

  late final load = _load;
  /**
   * Load model artifacts via HTTP request(s).
   *
   * See the documentation to `tf.io.http` for details on the saved
   * artifacts.
   *
   * @returns The loaded model artifacts (if loading succeeds).
   */
  Future<ModelArtifacts> _load() async {
    final modelConfigRequest =
        await this.fetch(Uri.parse(this.path), this.requestInit);

    if (modelConfigRequest.statusCode >= 300) {
      throw Exception('Request to ${this.path} failed with status code ' +
          '${modelConfigRequest.statusCode}. Please verify this URL points to ' +
          'the model JSON of the model to load.');
    }
    final ModelJSON modelJSON;
    try {
      final response = await http.Response.fromStream(modelConfigRequest);
      final bodyJsonMap = jsonDecode(response.body) as Map<String, Object?>;
      modelJSON = ModelJSON.fromJson(bodyJsonMap);
    } catch (e) {
      String message =
          'Failed to parse model JSON of response from ${this.path}.';
      // TODO(nsthorat): Remove this after some time when we're comfortable that
      // .pb files are mostly gone.
      if (this.path.endsWith('.pb')) {
        message += ' Your path contains a .pb file extension. ' +
            'Support for .pb models have been removed in TensorFlow.js 1.0 ' +
            'in favor of .json models. You can re-convert your Python ' +
            'TensorFlow model using the TensorFlow.js 1.0 conversion scripts ' +
            'or you can convert your.pb models with the \'pb2json\'' +
            'NPM script in the tensorflow/tfjs-converter repository.';
      } else {
        message += ' Please make sure the server is serving valid ' +
            'JSON for this request.';
      }
      throw Exception(message);
    }

    // We do not allow both modelTopology and weightsManifest to be missing.
    final modelTopology = modelJSON.modelTopology;
    final weightsManifest = modelJSON.weightsManifest;
    if (modelTopology == null && weightsManifest == null) {
      throw Exception(
          'The JSON from HTTP path ${this.path} contains neither model ' +
              'topology or manifest for weights.');
    }

    return getModelArtifactsForJSON(
        modelJSON, (weightsManifest) => this._loadWeights(weightsManifest));
  }

  Future<EncodedWeights> _loadWeights(
      WeightsManifestConfig weightsManifest) async {
    final weightPath = this.path is List ? this.path[1] : this.path;
    final p = parseUrl(weightPath);
    final prefix = p.first;
    final suffix = p.second;
    final pathPrefix = this.weightPathPrefix ?? prefix;

    final List<WeightsManifestEntry> weightSpecs = [];
    for (final entry in weightsManifest) {
      weightSpecs.addAll(entry.weights);
    }

    final List<String> fetchURLs = [];
    final List<Future<String>> urlPromises = [];
    for (final weightsGroup in weightsManifest) {
      for (final path in weightsGroup.paths) {
        if (this.weightUrlConverter != null) {
          urlPromises.add(this.weightUrlConverter!(path));
        } else {
          fetchURLs.add(pathPrefix + path + suffix);
        }
      }
    }

    if (this.weightUrlConverter != null) {
      fetchURLs.addAll(await Future.wait(urlPromises));
    }

    final buffers = await loadWeightsAsArrayBuffer(
      fetchURLs,
      LoadOptions(
        requestInit: this.requestInit,
        fetchFunc: this.fetch,
        onProgress: this.onProgress,
      ),
    );
    return EncodedWeights(
      specs: weightSpecs,
      data: concatenateArrayBuffers(buffers),
    );
  }
}

/**
 * Extract the prefix and suffix of the url, where the prefix is the path before
 * the last file, and suffix is the search params after the last file.
 * ```
 * const url = 'http://tfhub.dev/model/1/tensorflowjs_model.pb?tfjs-format=file'
 * [prefix, suffix] = parseUrl(url)
 * // prefix = 'http://tfhub.dev/model/1/'
 * // suffix = '?tfjs-format=file'
 * ```
 * @param url the model url to be parsed.
 */
Tuple<String, String> parseUrl(String url) {
  final lastSlash = url.lastIndexOf('/');
  final lastSearchParam = url.lastIndexOf('?');
  final prefix = url.substring(0, lastSlash);
  final suffix =
      lastSearchParam > lastSlash ? url.substring(lastSearchParam) : '';
  return Tuple(prefix + '/', suffix);
}

bool isHTTPScheme(String url) {
  return HTTPRequest.URL_SCHEME_REGEX.hasMatch(url);
}

IOHandler? httpRouter(List<String> url, LoadOptions? loadOptions) {
  if (
      // TODO: typeof fetch == 'undefined'
      false && (loadOptions == null || loadOptions.fetchFunc == null)) {
    // `http` uses `fetch` or `node-fetch`, if one wants to use it in
    // an environment that is not the browser or node they have to setup a
    // global fetch polyfill.
    return null;
  } else {
    bool isHTTP = true;
    if (url is List) {
      isHTTP = url.every((urlItem) => isHTTPScheme(urlItem));
    } else {
      isHTTP = isHTTPScheme(url as String);
    }
    if (isHTTP) {
      // TODO: was the entire list
      return httpHandler(url.first, loadOptions);
    }
  }
  return null;
}

/**
 * Creates an IOHandler subtype that sends model artifacts to HTTP server.
 *
 * An HTTP request of the `multipart/form-data` mime type will be sent to the
 * `path` URL. The form data includes artifacts that represent the topology
 * and/or weights of the model. In the case of Keras-style `tf.Model`, two
 * blobs (files) exist in form-data:
 *   - A JSON file consisting of `modelTopology` and `weightsManifest`.
 *   - A binary weights file consisting of the concatenated weight values.
 * These files are in the same format as the one generated by
 * [tfjs_converter](https://js.tensorflow.org/tutorials/import-keras.html).
 *
 * The following code snippet exemplifies the client-side code that uses this
 * function:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save(tf.io.http(
 *     'http://model-server:5000/upload', {requestInit: {method: 'PUT'}}));
 * console.log(saveResult);
 * ```
 *
 * If the default `POST` method is to be used, without any custom parameters
 * such as headers, you can simply pass an HTTP or HTTPS URL to `model.save`:
 *
 * ```js
 * const saveResult = await model.save('http://model-server:5000/upload');
 * ```
 *
 * The following GitHub Gist
 * https://gist.github.com/dsmilkov/1b6046fd6132d7408d5257b0976f7864
 * implements a server based on [flask](https://github.com/pallets/flask) that
 * can receive the request. Upon receiving the model artifacts via the requst,
 * this particular server reconsistutes instances of [Keras
 * Models](https://keras.io/models/model/) in memory.
 *
 *
 * @param path A URL path to the model.
 *   Can be an absolute HTTP path (e.g.,
 *   'http://localhost:8000/model-upload)') or a relative path (e.g.,
 *   './model-upload').
 * @param requestInit Request configurations to be used when sending
 *    HTTP request to server using `fetch`. It can contain fields such as
 *    `method`, `credentials`, `headers`, `mode`, etc. See
 *    https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
 *    for more information. `requestInit` must not have a body, because the
 * body will be set by TensorFlow.js. File blobs representing the model
 * topology (filename: 'model.json') and the weights of the model (filename:
 * 'model.weights.bin') will be appended to the body. If `requestInit` has a
 * `body`, an Error will be thrown.
 * @param loadOptions Optional configuration for the loading. It includes the
 *   following fields:
 *   - weightPathPrefix Optional, this specifies the path prefix for weight
 *     files, by default this is calculated from the path param.
 *   - fetchFunc Optional, custom `fetch` function. E.g., in Node.js,
 *     the `fetch` from node-fetch can be used here.
 *   - onProgress Optional, progress callback function, fired periodically
 *     before the load is completed.
 * @returns An instance of `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
IOHandler httpHandler(String path, LoadOptions? loadOptions) {
  return HTTPRequest(path, loadOptions);
}

/**
 * Deprecated. Use `tf.io.http`.
 * @param path
 * @param loadOptions
 */
// IOHandler browserHTTPRequest(String path, LoadOptions? loadOptions) {
//   return httpHandler(path, loadOptions);
// }