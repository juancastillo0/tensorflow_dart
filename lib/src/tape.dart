import 'package:tensorflow_wasm/src/engine.dart';
import 'package:tensorflow_wasm/src/tensor.dart';
import 'package:tensorflow_wasm/src/util_base.dart' as util;

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

// import {Tensor} from './tensor';
// import {NamedTensorMap} from './tensor_types';
// import * as util from './util';

class TapeNode {
  final int id;
  final String kernelName;
  final List<Tensor> outputs;
  final NamedTensorMap inputs;
  // Optional params, defined only for ops with gradient impl.
  NamedGradientMap Function(List<Tensor> dys)? gradient;
  final List<Tensor>? saved;

  TapeNode({
    required this.id,
    required this.kernelName,
    required this.outputs,
    required this.inputs,
    this.gradient,
    this.saved,
  });

  TapeNode copyWith({
    int? id,
    String? kernelName,
    List<Tensor>? outputs,
    NamedTensorMap? inputs,
  }) {
    return TapeNode(
      id: id ?? this.id,
      kernelName: kernelName ?? this.kernelName,
      outputs: outputs ?? this.outputs,
      inputs: inputs ?? this.inputs,
      gradient: this.gradient,
      saved: this.saved,
    );
  }
}

typedef NamedGradientMap = Map<String, Tensor Function()>;

/**
 * Computes a list of TapeNodes that connect x to y, filtering everything else
 * out and preserving the order of the original tape elements.
 *
 * @param tape The tape elements to filter.
 * @param xs The input Tensors.
 * @param y The output Tensor.
 */
List<TapeNode> getFilteredNodesXToY(
    List<TapeNode> tape, List<Tensor> xs, Tensor y) {
  // Forward pass to compute all the nodes and Tensors that are transitively a
  // function of x.
  final tensorsFromX = <int, bool>{};
  final nodesFromX = <int, bool>{};
  for (int i = 0; i < xs.length; i++) {
    tensorsFromX[xs[i].id] = true;
  }

  for (int i = 0; i < tape.length; i++) {
    final node = tape[i];
    final nodeInputs = node.inputs;
    for (final inputName in nodeInputs.keys) {
      final input = nodeInputs[inputName];

      bool anyInputFromX = false;
      for (int j = 0; j < xs.length; j++) {
        if (tensorsFromX[input?.id] == true) {
          node.outputs.forEach((output) => tensorsFromX[output.id] = true);
          anyInputFromX = true;
          nodesFromX[node.id] = true;
          break;
        }
      }

      if (anyInputFromX) {
        break;
      }
    }
  }

  // Backward pass to find all of the nodes and Tensors that lead to y.
  final tensorsLeadToY = <int, bool>{};
  tensorsLeadToY[y.id] = true;
  final nodesToY = <int, bool>{};

  for (int i = tape.length - 1; i >= 0; i--) {
    final node = tape[i];
    final nodeInputs = node.inputs;

    // If any of the outputs lead to y, mark all of the inputs as leading to y.
    for (int j = 0; j < node.outputs.length; j++) {
      if (tensorsLeadToY[node.outputs[j].id] == true) {
        for (final inputName in nodeInputs.keys) {
          tensorsLeadToY[nodeInputs[inputName]!.id] = true;
          nodesToY[node.id] = true;
        }
        break;
      }
    }
  }

  // Return the paths that come from x and lead to y.
  final filteredTape = <TapeNode>[];
  for (int i = 0; i < tape.length; i++) {
    final node = tape[i];

    if (nodesFromX[node.id] == true && nodesToY[node.id] == true) {
      // Prune the inputs from the node that aren't a function of x.
      final prunedInputs = <String, Tensor>{};
      for (final inputName in node.inputs.keys) {
        final nodeInput = node.inputs[inputName]!;
        if (tensorsFromX[nodeInput.id] == true) {
          prunedInputs[inputName] = nodeInput;
        }
      }

      // Copy the node and overwrite inputsAndArgs to the pruned version.
      final prunedNode = node.copyWith(inputs: prunedInputs);

      filteredTape.add(prunedNode);
    }
  }

  return filteredTape;
}

/**
 * Backpropagate gradients through the filtered TapeNodes.
 *
 * @param tensorAccumulatedGradientMap A map of Tensor to its gradient. This map
 * is mutated by this method.
 * @param filteredTape The filtered TapeNodes to backprop through.
 */
void backpropagateGradients(
    Map<int, Tensor> tensorAccumulatedGradientMap,
    List<TapeNode> filteredTape,
    Tensor Function(Function f) tidy,
    Tensor Function(Tensor, Tensor) add) {
  // Walk the tape backward and keep a map of Tensor to its gradient.
  for (int i = filteredTape.length - 1; i >= 0; i--) {
    final node = filteredTape[i];

    final dys = <Tensor?>[]; // TODO: wasn't null
    node.outputs.forEach((o) {
      final gradTensor = tensorAccumulatedGradientMap[o.id];
      if (gradTensor != null) {
        dys.add(gradTensor);
      } else {
        // This particular output is not in the back-propagation subgraph, so it
        // does not affect the final output, thus we put null for its dy.
        dys.add(null);
      }
    });

    if (node.gradient == null) {
      throw Exception('Cannot compute gradient: gradient function not found ' +
          'for ${node.kernelName}.');
    }

    // Backprop dy through this node and accumulate gradients over the inputs.
    final inputGradients = node.gradient!(dys.cast()); // TODO:

    for (final inputName in node.inputs.keys) {
      if (!inputGradients.containsKey(inputName)) {
        throw Exception('Cannot backprop through input ${inputName}. ' +
            'Available gradients found: ${inputGradients.keys}.');
      }

      // Call the gradient function.
      final dx = tidy(() => inputGradients[inputName]!());
      if (dx.dtype != 'float32') {
        throw Exception(
            'Error in gradient for op ${node.kernelName}. The gradient of input ' +
                "${inputName} must have 'float32' dtype, but has '${dx.dtype}'");
      }
      final x = node.inputs[inputName]!;
      if (!util.arraysEqual(dx.shape, x.shape)) {
        throw Exception(
            'Error in gradient for op ${node.kernelName}. The gradient of input ' +
                "'${inputName}' has shape '${dx.shape}', which does not match " +
                "the shape of the input '${x.shape}'");
      }

      final curGradient = tensorAccumulatedGradientMap[x.id];
      if (curGradient == null) {
        tensorAccumulatedGradientMap[x.id] = dx;
      } else {
        tensorAccumulatedGradientMap[x.id] = add(curGradient, dx);
        curGradient.dispose();
      }
    }
  }
}
