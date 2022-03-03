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

// import * as tf from '@tensorflow/tfjs';
// import * as fs from 'fs';
// import {dirname, join, resolve} from 'path';
// import {promisify} from 'util';
// import {getModelArtifactsInfoForJSON, toArrayBuffer} from './io_utils';

// const stat = promisify(fs.stat);
// const writeFile = promisify(fs.writeFile);
// const readFile = promisify(fs.readFile);
// const mkdir = promisify(fs.mkdir);

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:path/path.dart' as p;
import '../util_base.dart' as util;

import 'io.dart';

Future<Uint8List> readFile(String path) {
  return File(path).readAsBytes();
}

Future<String> readFileString(String path) {
  return File(path).readAsString();
}

Never _doesNotExistHandler(String name, String path) {
  throw Exception('${name} ${path} does not exist: loading failed');
}

bool isComposedUrl(ModelUri uri) => uri.length >= 2;

class NodeFileSystem implements IOHandler {
  static const URL_SCHEME = 'file://';

  late final ModelUri path;

  static const MODEL_JSON_FILENAME = 'model.json';
  static const WEIGHTS_BINARY_FILENAME = 'weights.bin';
  static const MODEL_BINARY_FILENAME = 'tensorflowjs.pb';

  /**
   * Constructor of the NodeFileSystem IOHandler.
   * @param path A single path or an Array of paths.
   *   For saving: expects a single path pointing to an existing or nonexistent
   *     directory. If the directory does not exist, it will be
   *     created.
   *   For loading:
   *     - If the model has JSON topology (e.g., `tf.Model`), a single path
   *       pointing to the JSON file (usually named `model.json`) is expected.
   *       The JSON file is expected to contain `modelTopology` and/or
   *       `weightsManifest`. If `weightManifest` exists, the values of the
   *       weights will be loaded from relative paths (relative to the directory
   *       of `model.json`) as contained in `weightManifest`.
   *     - If the model has binary (protocol buffer GraphDef) topology,
   *       an Array of two paths is expected: the first path should point to the
   *       .pb file and the second path should point to the weight manifest
   *       JSON file.
   */
  NodeFileSystem(ModelUri path) {
    if (isComposedUrl(path)) {
      util.assert_(
          path.length == 2,
          () =>
              'file paths must have a length of 2, ' +
              '(actual length is ${path.length}).');
      this.path = path.map((p_) => p.canonicalize(p_)).toList();
    }
    //  else {
    //   this.path = p.canonicalize(path);
    // }
  }

  late final save = _save;
  Future<SaveResult> _save(ModelArtifacts modelArtifacts) async {
    if (isComposedUrl(this.path)) {
      throw Exception('Cannot perform saving to multiple paths.');
    }

    await this.createOrVerifyDirectory();

    if (modelArtifacts.modelTopology is ByteBuffer) {
      throw Exception(
          'NodeFileSystem.save() does not support saving model topology ' +
              'in binary format yet.');
      // TODO(cais, nkreeger): Implement this. See
      //   https://github.com/tensorflow/tfjs/issues/343
    } else {
      final weightsBinPath = p.join(this.path.first, WEIGHTS_BINARY_FILENAME);
      final weightsManifest = [
        WeightsManifestGroupConfig(
          paths: [WEIGHTS_BINARY_FILENAME],
          weights: modelArtifacts.weightSpecs!,
        )
      ];
      final modelJSON = ModelJSON(
        modelTopology: modelArtifacts.modelTopology as Map,
        weightsManifest: weightsManifest,
        format: modelArtifacts.format,
        generatedBy: modelArtifacts.generatedBy,
        convertedBy: modelArtifacts.convertedBy,
        trainingConfig: modelArtifacts.trainingConfig,
        signature: modelArtifacts.signature,
        userDefinedMetadata: modelArtifacts.userDefinedMetadata,
      );

      final modelJSONPath = p.join(this.path.first, MODEL_JSON_FILENAME);
      await File(modelJSONPath).writeAsString(jsonEncode(modelJSON));
      await File(weightsBinPath)
          .writeAsBytes(modelArtifacts.weightData!.asUint8List());

      return SaveResult(
        // TODO(cais): Use explicit ModelArtifactsInfo type below once it
        // is available.
        // tslint:disable-next-line:no-any
        modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts),
      );
    }
  }

  late final load = _load;
  Future<ModelArtifacts> _load() async {
    return isComposedUrl(this.path)
        ? this.loadBinaryModel()
        : this.loadJSONModel();
  }

  Future<ModelArtifacts> loadBinaryModel() async {
    final topologyPath = this.path[0];
    final weightManifestPath = this.path[1];
    final topology = await FileStat.stat(topologyPath);
    final weightManifest = await FileStat.stat(weightManifestPath);

    // `this.path` can be either a directory or a file. If it is a file, assume
    // it is model.json file.
    if (topology.type != FileSystemEntityType.file) {
      if (topology.type == FileSystemEntityType.notFound) {
        _doesNotExistHandler('Topology Path', topologyPath);
      }
      throw Exception('File specified for topology is not a file!');
    }
    if (weightManifest.type != FileSystemEntityType.file) {
      if (weightManifest.type == FileSystemEntityType.notFound) {
        _doesNotExistHandler('Weight Manifest Path', weightManifestPath);
      }
      throw Exception('File specified for the weight manifest is not a file!');
    }

    final modelTopology = await readFile(this.path[0]);
    final weightsManifest = jsonDecode(await readFileString(this.path[1]));

    final weight = await this._loadWeights(weightsManifest, this.path[1]);

    final modelArtifacts = ModelArtifacts(
      modelTopology: modelTopology,
      weightSpecs: weight.specs,
      weightData: weight.data,
    );

    return modelArtifacts;
  }

  Future<ModelArtifacts> loadJSONModel() async {
    final path = this.path as String;
    final info = await FileStat.stat(path);

    // `path` can be either a directory or a file. If it is a file, assume
    // it is model.json file.
    if (info.type == FileSystemEntityType.file) {
      final modelJSON = jsonDecode(await readFileString(path));
      return getModelArtifactsForJSON(ModelJSON.fromJson(modelJSON),
          (weightsManifest) => this._loadWeights(weightsManifest, path));
    } else {
      if (info.type == FileSystemEntityType.notFound) {
        _doesNotExistHandler('Path', path);
      }
      throw Exception(
          'The path to load from must be a file. Loading from a directory ' +
              'is not supported.');
    }
  }

  Future<EncodedWeights> _loadWeights(
    WeightsManifestConfig weightsManifest,
    String path,
  ) async {
    final dirName = p.dirname(path);
    final buffers = <Uint8List>[];
    final weightSpecs = <WeightsManifestEntry>[];
    for (final group in weightsManifest) {
      for (final path in group.paths) {
        final weightFilePath = p.join(dirName, path);

        final buffer = await readFile(weightFilePath);
        buffers.add(buffer);
      }
      weightSpecs.addAll(group.weights);
    }
    return EncodedWeights(specs: weightSpecs, data: toArrayBuffer(buffers));
  }

  /**
   * For each item in `this.path`, creates a directory at the path or verify
   * that the path exists as a directory.
   */
  Future<void> createOrVerifyDirectory() async {
    final paths =
        this.path; // isComposedUrl(this.path) ? this.path : [this.path];
    for (final path in paths) {
      try {
        await Directory(path).create(recursive: true);
      } catch (e) {
        if (e is FileSystemException) {
          if ((await FileStat.stat(path)).type == FileSystemEntityType.file) {
            throw Exception('Path ${path} exists as a file. The path must be ' +
                'nonexistent or point to a directory.');
          }
          // else continue, the directory exists
        } else {
          rethrow;
        }
      }
    }
  }
}

NodeFileSystem? nodeFileSystemRouter(ModelUri url, LoadOptions? _) {
  if (isComposedUrl(url)) {
    if (url.every(
        (urlElement) => urlElement.startsWith(NodeFileSystem.URL_SCHEME))) {
      return NodeFileSystem(url
          .map((urlElement) =>
              urlElement.substring(NodeFileSystem.URL_SCHEME.length))
          .toList());
    } else {
      return null;
    }
  } else {
    if (url.first.startsWith(NodeFileSystem.URL_SCHEME)) {
      return NodeFileSystem(
          [url.first.substring(NodeFileSystem.URL_SCHEME.length)]);
    } else {
      return null;
    }
  }
}
// Registration of `nodeFileSystemRouter` is done in index.ts.

/**
 * Factory function for Node.js native file system IO Handler.
 *
 * @param path A single path or an Array of paths.
 *   For saving: expects a single path pointing to an existing or nonexistent
 *     directory. If the directory does not exist, it will be
 *     created.
 *   For loading:
 *     - If the model has JSON topology (e.g., `tf.Model`), a single path
 *       pointing to the JSON file (usually named `model.json`) is expected.
 *       The JSON file is expected to contain `modelTopology` and/or
 *       `weightsManifest`. If `weightManifest` exists, the values of the
 *       weights will be loaded from relative paths (relative to the directory
 *       of `model.json`) as contained in `weightManifest`.
 *     - If the model has binary (protocol buffer GraphDef) topology,
 *       an Array of two paths is expected: the first path should point to the
 *        .pb file and the second path should point to the weight manifest
 *       JSON file.
 */
NodeFileSystem fileSystem(ModelUri path) {
  return NodeFileSystem(path);
}
