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

// Importing local_storage and indexed_db is necessary for the routers to be
// registered.

export 'router_registry.dart';
export 'types.dart';
export 'http.dart';
export 'weights_loader.dart';
export 'io_utils.dart';
export 'browser_files.dart';
export 'indexed_db.dart';
export 'file_system.dart';
export 'passthrough.dart';

import 'io.dart';

void setUpIo() {
  IORouterRegistry.registerSaveRouter(browserDownloadsRouter);
  IORouterRegistry.registerSaveRouter(nodeFileSystemRouter);
  IORouterRegistry.registerLoadRouter(nodeFileSystemRouter);
  IORouterRegistry.registerSaveRouter(httpRouter);
  IORouterRegistry.registerLoadRouter(httpRouter);
  IORouterRegistry.registerSaveRouter(indexedDBRouter);
  IORouterRegistry.registerLoadRouter(indexedDBRouter);
}

typedef ModelUri = List<String>; // string|string[]

class ModelUrl {
  const ModelUrl._();
  const factory ModelUrl.single(String path) = SingleModelUrl;

  const factory ModelUrl.combined({
    required String topologyPath,
    required String weightManifestPath,
  }) = CombinedModelUrl;

  bool get isSingle => this is SingleModelUrl;

  T when<T>({
    required T Function(String path) path,
    required T Function(CombinedModelUrl path) paths,
  }) {
    final v = this;
    if (v is CombinedModelUrl) return paths(v);
    return path((v as SingleModelUrl).path);
  }
}

class CombinedModelUrl extends ModelUrl {
  final String topologyPath;
  final String weightManifestPath;

  const CombinedModelUrl({
    required this.topologyPath,
    required this.weightManifestPath,
  }) : super._();
}

class SingleModelUrl extends ModelUrl {
  final String path;

  const SingleModelUrl(this.path) : super._();
}

/*
import './indexed_db';
import './local_storage';

import {browserFiles} from './browser_files';
import {browserHTTPRequest, http, isHTTPScheme} from './http';
import {concatenateArrayBuffers, decodeWeights, encodeWeights, getModelArtifactsForJSON, getModelArtifactsInfoForJSON} from './io_utils';
import {fromMemory, withSaveHandler} from './passthrough';
import {getLoadHandlers, getSaveHandlers, registerLoadRouter, registerSaveRouter} from './router_registry';
import {IOHandler, LoadHandler, LoadOptions, ModelArtifacts, ModelArtifactsInfo, ModelJSON, ModelStoreManager, OnProgressCallback, RequestDetails, SaveConfig, SaveHandler, SaveResult, TrainingConfig, WeightGroup, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {loadWeights, weightsLoaderFactory} from './weights_loader';

export {copyModel, listModels, moveModel, removeModel} from './model_management';
export {
  browserFiles,
  browserHTTPRequest,
  concatenateArrayBuffers,
  decodeWeights,
  encodeWeights,
  fromMemory,
  getLoadHandlers,
  getModelArtifactsForJSON,
  getModelArtifactsInfoForJSON,
  getSaveHandlers,
  http,
  IOHandler,
  isHTTPScheme,
  LoadHandler,
  LoadOptions,
  loadWeights,
  ModelArtifacts,
  ModelArtifactsInfo,
  ModelJSON,
  ModelStoreManager,
  OnProgressCallback,
  registerLoadRouter,
  registerSaveRouter,
  RequestDetails,
  SaveConfig,
  SaveHandler,
  SaveResult,
  TrainingConfig,
  WeightGroup,
  weightsLoaderFactory,
  WeightsManifestConfig,
  WeightsManifestEntry,
  withSaveHandler
};
*/