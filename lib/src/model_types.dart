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

import 'package:tensorflow_wasm/src/tensor.dart';

// import {Tensor} from './tensor';
// import {NamedTensorMap} from './tensor_types';
// import {DataType} from './types';

class ModelPredictConfig {
  /**
   * Optional. Batch size (Integer). If unspecified, it will default to 32.
   */
  final int? batchSize;

  /**
   * Optional. Verbosity mode. Defaults to false.
   */
  final bool? verbose;

  ModelPredictConfig({
    this.batchSize,
    this.verbose,
  });
}

/**
 * Interface for model input/output tensor info.
 */
class ModelTensorInfo {
  // Name of the tensor.
  final String name;
  // Tensor shape information, Optional.
  final List<int>? shape;
  // Data type of the tensor.
  final DataType dtype;
  // TensorFlow native Data type of the tensor.
  final String? tfDtype;

  ModelTensorInfo({
    required this.name,
    this.shape,
    required this.dtype,
    this.tfDtype,
  });

  @override
  String toString() {
    return 'ModelTensorInfo(name: $name, shape: $shape, dtype: $dtype, tfDtype: $tfDtype)';
  }
}

/**
 * Common interface for a machine learning model that can do inference.
 */
abstract class InferenceModel {
  /**
   * Return the array of input tensor info.
   */
  List<ModelTensorInfo> get inputs;

  /**
   * Return the array of output tensor info.
   */
  List<ModelTensorInfo> get outputs;

  /**
   * Execute the inference for the input tensors.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a Tensor. For models with multiple inputs, inputs
   * params should be in either Tensor[] if the input order is fixed, or
   * otherwise NamedTensorMap format.
   * For batch inference execution, the tensors for each input need to be
   * concatenated together. For example with mobilenet, the required input shape
   * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
   * If we are provide a batched data of 100 images, the input tensor should be
   * in the shape of [100, 244, 244, 3].
   *
   * @param config Prediction configuration for specifying the batch size.
   *
   * @returns Inference result tensors. The output would be single Tensor if
   * model has single output node, otherwise Tensor[] or NamedTensorMap[] will
   * be returned for model with multiple outputs.
   */

  Tensors predict(
    // Tensor|Tensor[]|NamedTensorMap
    // : Tensor|Tensor[]|NamedTensorMap
    Tensors inputs,
    ModelPredictConfig config,
  );

  /**
   * Single Execute the inference for the input tensors and return activation
   * values for specified output node names without batching.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a Tensor. For models with multiple inputs, inputs
   * params should be in either Tensor[] if the input order is fixed, or
   * otherwise NamedTensorMap format.
   *
   * @param outputs string|string[]. List of output node names to retrieve
   * activation from.
   *
   * @returns Activation values for the output nodes result tensors. The return
   * type matches specified parameter outputs type. The output would be single
   * Tensor if single output is specified, otherwise Tensor[] for multiple
   * outputs.
   */
  Tensors execute(
    // Tensor|Tensor[]|NamedTensorMap
    Tensors inputs,
    // string|string[]
    List<String> outputs,
  );
}

// /**
//  * @deprecated Deprecated interface for SavedModel/GraphModel MetaGraph info.
//  *     User MetaGraph instead.
//  */
// export interface MetaGraphInfo {
//   tags: string[];
//   signatureDefs: SignatureDefInfo;
// }

// /**
//  * @deprecated Deprecated interface for SavedModel/GraphModel SignatureDef info.
//  *     User SignatureDef instead.
//  */
// export interface SignatureDefInfo {
//   [key: string]: {
//     inputs: {[key: string]: SavedModelTensorInfo};
//     outputs: {[key: string]: SavedModelTensorInfo};
//   };
// }

// /**
//  * @deprecated Deprecated interface for SavedModel/GraphModel signature
//  *     input/output Tensor info. User ModelTensorInfo instead.
//  */
// export interface SavedModelTensorInfo {
//   dtype: string;
//   shape: number[];
//   name: string;
// }

// /**
//  * Interface for SavedModel/GraphModel MetaGraph info.
//  */
// export interface MetaGraph {
//   tags: string[];
//   signatureDefs: SignatureDef;
// }

// /**
//  * Interface for SavedModel/GraphModel SignatureDef entry.
//  */
// export interface SignatureDefEntry {
//   inputs: {[key: string]: ModelTensorInfo};
//   outputs: {[key: string]: ModelTensorInfo};
// }

// /**
//  * Interface for SavedModel/GraphModel SignatureDef info.
//  */
// export interface SignatureDef {
//   [key: string]: SignatureDefEntry;
// }