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

/* Type definitions for exporting and importing of models. */

import 'dart:typed_data';

import 'package:tensorflow_wasm/src/environment.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:http/http.dart' as http;

/**
 * A map from Tensor dtype to number of bytes per element of the Tensor.
 */
const DTYPE_VALUE_SIZE_MAP = {
  'float32': 4,
  'float16': 2,
  'int32': 4,
  'uint16': 2,
  'uint8': 1,
  'bool': 1,
  'complex64': 8
};

/**
 * A weight manifest.
 *
 * The weight manifest consists of an ordered list of weight-manifest groups.
 * Each weight-manifest group ("group" for short hereafter) consists of a
 * number of weight values stored in a number of paths.
 * See the documentation of `WeightManifestGroupConfig` below for more details.
 */
typedef WeightsManifestConfig = List<WeightsManifestGroupConfig>;

/**
 * A weight-manifest group.
 *
 * Consists of an ordered list of weight values encoded in binary format,
 * stored in an ordered list of paths.
 */
class WeightsManifestGroupConfig {
  /**
   * An ordered list of paths.
   *
   * Paths are intentionally abstract in order to be general. For example, they
   * can be relative URL paths or relative paths on the file system.
   */
  final List<String> paths;

  /**
   * Specifications of the weights stored in the paths.
   */
  final List<WeightsManifestEntry> weights;

  WeightsManifestGroupConfig({
    required this.paths,
    required this.weights,
  });

  Map<String, dynamic> toJson() {
    return {
      'paths': paths,
      'weights': weights.map((x) => x.toJson()).toList(),
    };
  }

  factory WeightsManifestGroupConfig.fromJson(Map<String, dynamic> map) {
    return WeightsManifestGroupConfig(
      paths: List<String>.from(map['paths']),
      weights: List<WeightsManifestEntry>.from(
          map['weights']?.map((x) => WeightsManifestEntry.fromJson(x))),
    );
  }
}

class Quantization {
  final double? scale; // The scaling constant to multiply by.
  final double? min; // The (possibly nudged) minimum weight to add.
  final String
      dtype; // The dtype of the quantized weights. 'uint16'|'uint8'|'float16'

  Quantization({
    this.scale,
    this.min,
    required this.dtype,
  });

  Map<String, dynamic> toJson() {
    return {
      'scale': scale,
      'min': min,
      'dtype': dtype,
    };
  }

  factory Quantization.fromJson(Map<String, dynamic> map) {
    return Quantization(
      scale: (map['scale'] as num?)?.toDouble(),
      min: (map['min'] as num?)?.toDouble(),
      dtype: map['dtype'],
    );
  }
}

/**
 * Group to which the weight belongs.
 *
 * - 'optimizer': Weight from a stateful optimizer.
 */
enum WeightGroup {
  model,
  optimizer,
}

extension WeightGroupToJson on WeightGroup {
  String toJson() => name;
}

/**
 * An entry in the weight manifest.
 *
 * The entry contains specification of a weight.
 */
class WeightsManifestEntry {
  /**
   * Name of the weight, e.g., 'Dense_1/bias'
   */
  final String name;

  /**
   * Shape of the weight.
   */
  final List<int> shape;

  /**
   * Data type of the weight.
   */
  final DataType dtype;

  /**
   * Type of the weight.
   *
   * Optional.
   *
   * The value 'optimizer' indicates the weight belongs to an optimizer
   * (i.e., used only during model training and not during inference).
   */
  final WeightGroup? group;

  /**
   * Information for dequantization of the weight.
   */
  final Quantization? quantization;

  WeightsManifestEntry({
    required this.name,
    required this.shape,
    required this.dtype,
    this.group,
    this.quantization,
  });

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'shape': shape,
      'dtype': dtype,
      'group': group?.toJson(),
      'quantization': quantization?.toJson(),
    };
  }

  factory WeightsManifestEntry.fromJson(Map<String, dynamic> map) {
    return WeightsManifestEntry(
      name: map['name'] ?? '',
      shape: List<int>.from(map['shape']),
      dtype: map['dtype'],
      group:
          map['group'] != null ? WeightGroup.values.byName(map['group']) : null,
      quantization: map['quantization'] != null
          ? Quantization.fromJson(map['quantization'])
          : null,
    );
  }
}

/**
 * Options for saving a model.
 * @innamespace io
 */
class SaveConfig {
  /**
   * Whether to save only the trainable weights of the model, ignoring the
   * non-trainable ones.
   */
  final bool? trainableOnly;

  /**
   * Whether the optimizer will be saved (if exists).
   *
   * Default: `false`.
   */
  final bool? includeOptimizer;

  const SaveConfig({
    this.trainableOnly,
    this.includeOptimizer,
  });
}

/**
 * Result of a saving operation.
 */
class SaveResult {
  /**
   * Information about the model artifacts saved.
   */
  final ModelArtifactsInfo modelArtifactsInfo;

  /**
   * HTTP responses from the server that handled the model-saving request (if
   * any). This is applicable only to server-based saving routes.
   */
  final List<Object>? responses; // Response

  /**
   * Error messages and related data (if any).
   */
  final List<Object>? errors; // {}|string

  SaveResult({
    required this.modelArtifactsInfo,
    this.responses,
    this.errors,
  });
}

enum ModelTopologyType {
  JSON,
  GraphDef,
}

class ModelArtifactsInfo {
  /**
   * Timestamp for when the model is saved.
   */
  final DateTime dateSaved;

  /**
   * TODO (cais,yassogba) consider removing GraphDef as GraphDefs now
   * come in a JSON format and none of our IOHandlers support a non json
   * format. We could conder replacing this with 'Binary' if we want to
   * allow future handlers to save to non json formats (though they will
   * probably want more information than 'Binary').
   * Type of the model topology
   *
   * Type of the model topology
   *
   * Possible values:
   *   - JSON: JSON config (human-readable, e.g., Keras JSON).
   *   - GraphDef: TensorFlow
   *     [GraphDef](https://www.tensorflow.org/extend/tool_developers/#graphdef)
   *     protocol buffer (binary).
   */
  final ModelTopologyType modelTopologyType;

  /**
   * Size of model topology (Keras JSON or GraphDef), in bytes.
   */
  final int? modelTopologyBytes;

  /**
   * Size of weight specification or manifest, in bytes.
   */
  final int? weightSpecsBytes;

  /**
   * Size of weight value data, in bytes.
   */
  final int? weightDataBytes;

  ModelArtifactsInfo({
    required this.dateSaved,
    required this.modelTopologyType,
    this.modelTopologyBytes,
    this.weightSpecsBytes,
    this.weightDataBytes,
  });
}

/** Model training configuration. */
class TrainingConfig {
  // TODO(cais): Tighten the typing once keras spec is available to tfjs-core.
  // See
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tfjs-layers/blob/master/src/keras_format/training_config.ts
  /** Optimizer used for the model training. */
  final Map optimizer_config; // {}

  // TODO(cais): Tighten the typing once keras spec is available to tfjs-core.
  /** Loss function(s) for the model's output(s). */
  final Object loss; // string|string[]|{[key: string]: string}

  // TODO(cais): Tighten the typing once keras spec is available to tfjs-core.
  /** Metric function(s) for the model's output(s). */
  final Object? metrics; // string[]|{[key: string]: string}

  // TODO(cais): Tighten the typing once keras spec is available to tfjs-core.
  final List<String>? weighted_metrics;

  // TODO(cais): Tighten the typing once keras spec is available to tfjs-core.
  final String? sample_weight_mode;

  final Object? loss_weights;

  TrainingConfig({
    required this.optimizer_config,
    required this.loss,
    this.metrics,
    this.weighted_metrics,
    this.sample_weight_mode,
    this.loss_weights,
  });

  Map<String, dynamic> toJson() {
    return {
      'optimizer_config': optimizer_config,
      'loss': loss,
      'metrics': metrics,
      'weighted_metrics': weighted_metrics,
      'sample_weight_mode': sample_weight_mode,
      'loss_weights': loss_weights,
    };
  }

  factory TrainingConfig.fromJson(Map<String, dynamic> map) {
    return TrainingConfig(
      optimizer_config: map['optimizer_config'],
      loss: map['loss'],
      metrics: map['metrics'],
      weighted_metrics: map['weighted_metrics'] as List<String>?,
      sample_weight_mode: map['sample_weight_mode'],
      loss_weights: map['loss_weights'],
    );
  }
}

/**
 * The serialized artifacts of a model, including topology and weights.
 *
 * The `modelTopology`, `trainingConfig`, `weightSpecs` and `weightData` fields
 * of this interface are optional, in order to support topology- or weights-only
 * saving and loading.
 *
 * Note this interface is used internally in IOHandlers.  For the file format
 * written to disk as `model.json`, see `ModelJSON`.
 */
class ModelArtifacts {
  /**
   * Model topology.
   *
   * For Keras-style `tf.Model`s, this is a JSON object.
   * For TensorFlow-style models (e.g., `SavedModel`), this is the JSON
   * encoding of the `GraphDef` protocol buffer.
   */
  final Object? modelTopology; // TODO: {}|ArrayBuffer

  /**
   * Serialized configuration for the model's training.
   */
  final TrainingConfig? trainingConfig;

  /**
   * Weight specifications.
   *
   * This corresponds to the weightsData below.
   */
  final List<WeightsManifestEntry>? weightSpecs;

  /**
   * Binary buffer for all weight values concatenated in the order specified
   * by `weightSpecs`.
   */
  final ByteBuffer? weightData;

  /**
   * Hard-coded format name for models saved from TensorFlow.js or converted
   * by TensorFlow.js Converter.
   */
  final String? format;

  /**
   * What library is responsible for originally generating this artifact.
   *
   * Used for debugging purposes. E.g., 'TensorFlow.js v1.0.0'.
   */
  final String? generatedBy;

  /**
   * What library or tool is responsible for converting the original model
   * to this format, applicable only if the model is output by a converter.
   *
   * Used for debugging purposes.  E.g., 'TensorFlow.js Converter v1.0.0'.
   *
   * A value of `null` means the model artifacts are generated without any
   * conversion process (e.g., saved directly from a TensorFlow.js
   * `tf.LayersModel` instance.)
   */
  final String? convertedBy;

  /**
   * Inputs and outputs signature for saved model.
   */
  final Map? signature;

  /**
   * User-defined metadata about the model.
   */
  final Map<String, Map>? userDefinedMetadata;

  /**
   * Initializer for the model.
   */
  final Map? modelInitializer;

  ModelArtifacts({
    this.trainingConfig,
    this.modelTopology,
    this.weightSpecs,
    this.weightData,
    this.format,
    this.generatedBy,
    this.convertedBy,
    this.signature,
    this.userDefinedMetadata,
    this.modelInitializer,
  });
}

/**
 * The on-disk format of the `model.json` file.
 *
 * TF.js 1.0 always populates the optional fields when writing model.json.
 * Prior versions did not provide those fields.
 */
class ModelJSON {
  /**
   * Model topology.
   *
   * For Keras-style `tf.Model`s, this is a JSON object.
   * For TensorFlow-style models (e.g., `SavedModel`), this is the JSON
   * encoding of the `GraphDef` protocol buffer.
   */
  final Map modelTopology;

  /** Model training configuration. */
  final TrainingConfig? trainingConfig;

  /**
   * Weights manifest.
   *
   * The weights manifest consists of an ordered list of weight-manifest
   * groups. Each weight-manifest group consists of a number of weight values
   * stored in a number of paths. See the documentation of
   * `WeightsManifestConfig` for more details.
   */
  final WeightsManifestConfig weightsManifest;

  /**
   * Hard-coded format name for models saved from TensorFlow.js or converted
   * by TensorFlow.js Converter.
   */
  final String? format;

  /**
   * What library is responsible for originally generating this artifact.
   *
   * Used for debugging purposes. E.g., 'TensorFlow.js v1.0.0'.
   */
  final String? generatedBy;

  /**
   * What library or tool is responsible for converting the original model
   * to this format, applicable only if the model is output by a converter.
   *
   * Used for debugging purposes.  E.g., 'TensorFlow.js Converter v1.0.0'.
   *
   * A value of `null` means the model artifacts are generated without any
   * conversion process (e.g., saved directly from a TensorFlow.js
   * `tf.LayersModel` instance.)
   */
  final String? convertedBy;

  /**
   * Inputs and outputs signature for saved model.
   */
  final Map? signature;

  /**
   * User-defined metadata about the model.
   */
  final Map<String, Map>? userDefinedMetadata;

  /**
   * Initializer for the model.
   */
  final Map? modelInitializer;

  ModelJSON({
    required this.modelTopology,
    this.trainingConfig,
    required this.weightsManifest,
    this.format,
    this.generatedBy,
    this.convertedBy,
    this.signature,
    this.userDefinedMetadata,
    this.modelInitializer,
  });

  Map<String, dynamic> toJson() {
    return {
      'modelTopology': modelTopology,
      'trainingConfig': trainingConfig?.toJson(),
      'weightsManifest': weightsManifest.map((e) => e.toJson()).toList(),
      'format': format,
      'generatedBy': generatedBy,
      'convertedBy': convertedBy,
      'signature': signature,
      'userDefinedMetadata': userDefinedMetadata,
      'modelInitializer': modelInitializer,
    };
  }

  factory ModelJSON.fromJson(Map<String, dynamic> map) {
    return ModelJSON(
      modelTopology: map['modelTopology'],
      trainingConfig: map['trainingConfig'] != null
          ? TrainingConfig.fromJson(map['trainingConfig'])
          : null,
      weightsManifest: (map['weightsManifest'] as List)
          .map((e) => WeightsManifestGroupConfig.fromJson(e))
          .toList(),
      format: map['format'],
      generatedBy: map['generatedBy'],
      convertedBy: map['convertedBy'],
      signature: map['signature'],
      userDefinedMetadata: map['userDefinedMetadata'] == null
          ? null
          : Map<String, Map>.from(map['userDefinedMetadata']),
      modelInitializer: map['modelInitializer'],
    );
  }
}

/**
 * Type definition for handlers of loading operations.
 */
typedef LoadHandler = Future<ModelArtifacts> Function();

/**
 * Type definition for handlers of saving operations.
 */
typedef SaveHandler = Future<SaveResult> Function(ModelArtifacts modelArtifact);

/**
 * Interface for a model import/export handler.
 *
 * The `save` and `load` handlers are both optional, in order to allow handlers
 * that support only saving or loading.
 */
// tslint:disable-next-line:interface-name
abstract class IOHandler {
  SaveHandler? get save;
  LoadHandler? get load;

  factory IOHandler({
    SaveHandler? save,
    LoadHandler? load,
  }) = _IOHandler;
}

class _IOHandler implements IOHandler {
  final SaveHandler? save;
  final LoadHandler? load;

  _IOHandler({
    this.save,
    this.load,
  });
}

/**
 * An interface for the manager of a model store.
 *
 * A model store is defined as a storage medium on which multiple models can
 * be stored. Each stored model has a unique `path` as its identifier.
 * A `ModelStoreManager` for the store allows actions including
 *
 * - Listing the models stored in the store.
 * - Deleting a model from the store.
 */
abstract class ModelStoreManager {
  /**
   * List all models in the model store.
   *
   * @returns A dictionary mapping paths of existing models to their
   *   model artifacts info. Model artifacts info include type of the model's
   *   topology, byte sizes of the topology, weights, etc.
   */
  Future<Map<String, ModelArtifactsInfo>> listModels();

  /**
   * Remove a model specified by `path`.
   *
   * @param path
   * @returns ModelArtifactsInfo of the deleted model (if and only if deletion
   *   is successful).
   * @throws Error if deletion fails, e.g., if no model exists at `path`.
   */
  Future<ModelArtifactsInfo> removeModel(String path);
}

/**
 * Callback for the progress of a long-running action such as an HTTP
 * request for a large binary object.
 *
 * `fraction` should be a number in the [0, 1] interval, indicating how
 * much of the action has completed.
 */
typedef OnProgressCallback = void Function(double fraction);

/** @innamespace io */
class LoadOptions {
  /**
   * RequestInit (options) for HTTP requests.
   *
   * For detailed information on the supported fields, see
   * [https://developer.mozilla.org/en-US/docs/Web/API/Request/Request](
   *     https://developer.mozilla.org/en-US/docs/Web/API/Request/Request)
   */
  final RequestInit? requestInit; // TODO: RequestInit

  /**
   * Progress callback.
   */
  final OnProgressCallback? onProgress;

  /**
   * A function used to override the `window.fetch` function.
   */
  final FetchFn? fetchFunc;

  /**
   * Strict loading model: whether extraneous weights or missing
   * weights should trigger an `Error`.
   *
   * If `true`, require that the provided weights exactly match those
   * required by the layers. `false` means that both extra weights
   * and missing weights will be silently ignored.
   *
   * Default: `true`.
   */
  final bool? strict;

  /**
   * Path prefix for weight files, by default this is calculated from the
   * path of the model JSON file.
   *
   * For instance, if the path to the model JSON file is
   * `http://localhost/foo/model.json`, then the default path prefix will be
   * `http://localhost/foo/`. If a weight file has the path value
   * `group1-shard1of2` in the weight manifest, then the weight file will be
   * loaded from `http://localhost/foo/group1-shard1of2` by default. However,
   * if you provide a `weightPathPrefix` value of
   * `http://localhost/foo/alt-weights`, then the weight file will be loaded
   * from the path `http://localhost/foo/alt-weights/group1-shard1of2` instead.
   */
  final String? weightPathPrefix;

  /**
   * Whether the module or model is to be loaded from TF Hub.
   *
   * Setting this to `true` allows passing a TF-Hub module URL, omitting the
   * standard model file name and the query parameters.
   *
   * Default: `false`.
   */
  final bool? fromTFHub;

  /**
   * An async function to convert weight file name to URL. The weight file
   * names are stored in model.json's weightsManifest.paths field. By default we
   * consider weight files are colocated with the model.json file. For example:
   *     model.json URL: https://www.google.com/models/1/model.json
   *     group1-shard1of1.bin url:
   *        https://www.google.com/models/1/group1-shard1of1.bin
   *
   * With this func you can convert the weight file name to any URL.
   */
  final Future<String> Function(String weightFileName)? weightUrlConverter;

  const LoadOptions({
    this.requestInit,
    this.onProgress,
    this.fetchFunc,
    this.strict,
    this.weightPathPrefix,
    this.fromTFHub,
    this.weightUrlConverter,
  });
}

/**
 * Additional options for Platform.fetch
 */
abstract class RequestDetails {
  /**
   * Is this request for a binary file (as opposed to a json file)
   */
  bool? get isBinary;
}
