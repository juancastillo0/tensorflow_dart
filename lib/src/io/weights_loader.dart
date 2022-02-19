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

// import {env} from '../environment';

// import {NamedTensorMap} from '../tensor_types';
// import * as util from '../util';
// import {decodeWeights} from './io_utils';
// import {monitorPromisesProgress} from './progress';
// import {DTYPE_VALUE_SIZE_MAP, LoadOptions, WeightsManifestConfig, WeightsManifestEntry} from './types';

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/environment.dart';
import 'package:tensorflow_wasm/src/io/io_utils.dart';
import 'package:tensorflow_wasm/src/io/progress.dart';
import 'package:tensorflow_wasm/src/io/types.dart';
import 'package:tensorflow_wasm/tensorflow_wasm.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;
import 'package:collection/collection.dart';

/**
 * Reads binary weights data from a number of URLs.
 *
 * @param fetchURLs URLs to send the HTTP requests at, using `fetch` calls.
 * @param requestOptions RequestInit (options) for the HTTP requests.
 * @param fetchFunc Optional overriding value for the `window.fetch` function.
 * @param onProgress Optional, progress callback function, fired periodically
 *   before the load is completed.
 * @returns A `Promise` of an Array of `ArrayBuffer`. The Array has the same
 *   length as `fetchURLs`.
 */
Future<List<ByteBuffer>> loadWeightsAsArrayBuffer(
  List<String> fetchURLs, [
  LoadOptions loadOptions = const LoadOptions(),
]) async {
  final fetchFunc = loadOptions.fetchFunc ?? env().platform!.fetch;

  // Create the requests for all of the weights in parallel.
  final requests = fetchURLs.map((fetchURL) =>
      // TODO: {isBinary: true}
      fetchFunc(Uri.parse(fetchURL), loadOptions.requestInit)).toList();

  final fetchStartFraction = 0.0;
  final fetchEndFraction = 0.5;

  final responses = loadOptions.onProgress == null
      ? await Future.wait(requests)
      : await monitorPromisesProgress(requests, loadOptions.onProgress!,
          fetchStartFraction, fetchEndFraction);

  final bufferPromises =
      responses.map((response) => response.stream.toBytes()).toList();

  final bufferStartFraction = 0.5;
  final bufferEndFraction = 1.0;

  final buffers = loadOptions.onProgress == null
      ? await Future.wait(bufferPromises)
      : await monitorPromisesProgress(bufferPromises, loadOptions.onProgress!,
          bufferStartFraction, bufferEndFraction);
  return buffers.map((e) => e.buffer).toList();
}

/**
 * Reads a weights manifest JSON configuration, fetches the weights and
 * returns them as `Tensor`s.
 *
 * @param manifest The weights manifest JSON.
 * @param filePathPrefix The path prefix for filenames given in the manifest.
 *     Defaults to the empty string.
 * @param weightNames The names of the weights to be fetched.
 */
Future<NamedTensorMap> loadWeights(
  WeightsManifestConfig manifest, [
  String filePathPrefix = '',
  List<String>? weightNames,
  RequestInit? requestInit,
]) async {
  // TODO(nsthorat): Groups are currently fetched atomically. If you need a
  // single weight from a group, the whole group will be fetched. At a future
  // date, we should support fetching only the individual shards within a
  // group that are needed to reconstruct the requested weight.
  // TODO(cais): Use `decodeWeights` for implementation.

  Future<List<ByteBuffer>> fetchWeights(List<String> fetchUrls) =>
      loadWeightsAsArrayBuffer(
          fetchUrls, LoadOptions(requestInit: requestInit));
  final loadWeights = weightsLoaderFactory(fetchWeights);

  return loadWeights(manifest, filePathPrefix, weightNames);
}

class WeightToFetch {
  final WeightsManifestEntry manifestEntry;
  final int groupOffset;
  final int sizeBytes;

  WeightToFetch({
    required this.manifestEntry,
    required this.groupOffset,
    required this.sizeBytes,
  });
}

/**
 * Creates a function, which reads a weights manifest JSON configuration,
 * fetches the weight files using the specified function and returns them as
 * `Tensor`s.
 *
 * ```js
 * // example for creating a nodejs weight loader, which reads the weight files
 * // from disk using fs.readFileSync
 *
 * import * as fs from 'fs'
 *
 * const fetchWeightsFromDisk = (filePaths: string[]) =>
 *   filePaths.map(filePath => fs.readFileSync(filePath).buffer)
 *
 * const loadWeights = tf.io.weightsLoaderFactory(fetchWeightsFromDisk)
 *
 * const manifest = JSON.parse(
 *   fs.readFileSync('./my_model-weights_manifest').toString()
 * )
 * const weightMap = await loadWeights(manifest, './')
 * ```
 * @param fetchWeightsFunction The function used for fetching the weight files.
 * @returns Weight loading function.
 */
Future<NamedTensorMap> Function(
  WeightsManifestConfig manifest,
  String filePathPrefix,
  List<String>? weightNames,
) weightsLoaderFactory(
  Future<List<ByteBuffer>> Function(List<String> fetchUrls)
      fetchWeightsFunction,
) {
  return (manifest, filePathPrefix, weightNames) async {
    // Collect all the groups, weights, and their relative offsets to be
    // fetched.
    final groupIndicesToFetchMap = manifest.map((_) => false).toList();
    final Map<int, List<WeightToFetch>> groupWeightsToFetch = {};
    final weightsFound =
        weightNames != null ? weightNames.map((_) => false).toList() : [];
    final List<String> allManifestWeightNames = [];
    manifest.forEachIndexed((groupIndex, manifestGroupConfig) {
      int groupOffset = 0;
      manifestGroupConfig.weights.forEach((weightsEntry) {
        final rawDtype = weightsEntry.quantization?.dtype ?? weightsEntry.dtype;

        final weightsBytes = DTYPE_VALUE_SIZE_MAP[rawDtype]! *
            util.sizeFromShape(weightsEntry.shape);

        void enqueueWeightsForFetchingFn() {
          groupIndicesToFetchMap[groupIndex] = true;
          if (groupWeightsToFetch[groupIndex] == null) {
            groupWeightsToFetch[groupIndex] = [];
          }

          groupWeightsToFetch[groupIndex]!.add(
            WeightToFetch(
                manifestEntry: weightsEntry,
                groupOffset: groupOffset,
                sizeBytes: weightsBytes),
          );
        }

        if (weightNames != null) {
          weightNames.forEachIndexed((weightIndex, weightName) {
            if (weightName == weightsEntry.name) {
              enqueueWeightsForFetchingFn();
              weightsFound[weightIndex] = true;
            }
          });
        } else {
          enqueueWeightsForFetchingFn();
        }

        allManifestWeightNames.add(weightsEntry.name);
        groupOffset += weightsBytes;
      });
    });

    if (!weightsFound.every((found) => found)) {
      int __i = 0;
      final weightsNotFound = weightNames!.where((_) => !weightsFound[__i++]);
      throw Exception("Could not find weights in manifest with names: " +
          "${weightsNotFound.join(', ')}. \n" +
          "Manifest JSON has weights with names: " +
          "${allManifestWeightNames.join(', ')}.");
    }

    // Convert the one-hot boolean groupId => shouldFetch map to a list of group
    // IDs.
    int __i = 0;
    final groupIndicesToFetch =
        groupIndicesToFetchMap.fold<List<int>>([], (accumulator, shouldFetch) {
      if (shouldFetch) {
        accumulator.add(__i);
      }
      __i++;
      return accumulator;
    });

    final List<String> fetchUrls = [];
    groupIndicesToFetch.forEach((i) {
      manifest[i].paths.forEach((filepath) {
        final fetchUrl = filePathPrefix +
            (!filePathPrefix.endsWith('/') ? '/' : '') +
            filepath;
        fetchUrls.add(fetchUrl);
      });
    });
    final buffers = await fetchWeightsFunction(fetchUrls);

    final NamedTensorMap weightsTensorMap = {};
    int bufferIndexOffset = 0;
    groupIndicesToFetch.forEach((i) {
      final numBuffers = manifest[i].paths.length;

      int groupBytes = 0;
      for (int i = 0; i < numBuffers; i++) {
        groupBytes += buffers[bufferIndexOffset + i].lengthInBytes;
      }

      // Create a buffer for the whole group.
      final groupBuffer = ByteData(groupBytes);
      final groupByteBuffer = Uint8List.view(groupBuffer.buffer);
      int groupBufferOffset = 0;
      for (int i = 0; i < numBuffers; i++) {
        final buffer = Uint8List.view(buffers[bufferIndexOffset + i]);
        List.copyRange(groupByteBuffer, groupBufferOffset,
            buffer); // groupByteBuffer.set(buffer, groupBufferOffset);
        groupBufferOffset += buffer.lengthInBytes;
      }

      final weightsEntries = groupWeightsToFetch[i]!;
      weightsEntries.forEach((weightsEntry) {
        final byteBuffer = groupBuffer.buffer.slice(
          weightsEntry.groupOffset,
          weightsEntry.groupOffset + weightsEntry.sizeBytes,
        );
        final nameToTensorMap =
            decodeWeights(byteBuffer, [weightsEntry.manifestEntry]);
        for (final e in nameToTensorMap.entries) {
          weightsTensorMap[e.key] = e.value;
        }
      });

      bufferIndexOffset += numBuffers;
    });

    return weightsTensorMap;
  };
}
