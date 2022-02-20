/**
 * Gather slices from tensor `x`'s axis `axis` according to `indices`.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const indices = tf.tensor1d([1, 3, 3], 'int32');
 *
 * x.gather(indices).print();
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const indices = tf.tensor1d([1, 1, 0], 'int32');
 *
 * x.gather(indices).print();
 * ```
 * @param x The input tensor whose slices to be gathered.
 * @param indices The indices of the values to extract.
 * @param axis The axis over which to select values. Defaults to 0.
 * @param batchDims Optional. The number of batch dimensions. It must be less
 *     than or equal to rank(indices). Defaults to 0.
 *     The output tensor will have shape of
 *     `x.shape[:axis] + indices.shape[batchDims:] + x.shape[axis + 1:]`
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function gather_<T extends Tensor>(
    x: T|TensorLike, indices: Tensor|TensorLike, axis = 0, batchDims = 0): T {
  const $x = convertToTensor(x, 'x', 'gather');
  const $indices = convertToTensor(indices, 'indices', 'gather', 'int32');

  const inputs: GatherV2Inputs = {x: $x, indices: $indices};
  const attrs: GatherV2Attrs = {axis, batchDims};

  return ENGINE.runKernel(
      GatherV2, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

/**
 * Gather slices from input tensor into a Tensor with shape specified by
 * `indices`.
 *
 * `indices` is an K-dimensional integer tensor, best thought of as a
 * (K-1)-dimensional tensor of indices into input, where each element defines a
 * slice of input:
 * output[\\(i_0, ..., i_{K-2}\\)] = input[indices[\\(i_0, ..., i_{K-2}\\)]]
 *
 * Whereas in `tf.gather`, `indices` defines slices into the first dimension of
 * input, in `tf.gatherND`, `indices` defines slices into the first N dimensions
 * of input, where N = indices.shape[-1].
 *
 * The last dimension of indices can be at most the rank of input:
 * indices.shape[-1] <= input.rank
 *
 * The last dimension of `indices` corresponds to elements
 * (if indices.shape[-1] == input.rank) or slices
 * (if indices.shape[-1] < input.rank) along dimension indices.shape[-1] of
 * input.
 * The output tensor has shape
 * indices.shape[:-1] + input.shape[indices.shape[-1]:]
 *
 * Note that on CPU, if an out of bound index is found, an error is returned. On
 * GPU, if an out of bound index is found, a 0 is stored in the corresponding
 * output value.
 *
 * ```js
 * const indices = tf.tensor2d([0, 1, 1, 0], [2,2], 'int32');
 * const input = tf.tensor2d([9, 10, 11, 12], [2, 2]);
 * tf.gatherND(input, indices).print() // [10, 11]
 * ```
 *
 * @param x The tensor from which to gather values.
 * @param indices Index tensor, must be of type int32.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function gatherND_(x: Tensor|TensorLike, indices: Tensor|TensorLike): Tensor {
  const $indices = convertToTensor(indices, 'indices', 'gatherND', 'int32');
  const $x = convertToTensor(x, 'x', 'gatherND', 'string_or_numeric');

  const inputs: GatherNdInputs = {params: $x, indices: $indices};

  return ENGINE.runKernel(GatherNd, inputs as {} as NamedTensorMap);
}

/**
 * Creates a new tensor by applying sparse updates to individual
 * values or slices within a zero tensor of the given shape tensor according to
 * indices. This operator is the inverse of the `tf.gatherND` operator which
 * extracts values or slices from a given tensor.
 *
 * ```js
 * const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
 * const updates = tf.tensor1d([9, 10, 11, 12]);
 * const shape = [8];
 * tf.scatterND(indices, updates, shape).print() //[0, 11, 0, 10, 9, 0, 0, 12]
 * ```
 *
 * @param indices The tensor contains the indices into the output tensor.
 * @param updates The tensor contains the value for the indices.
 * @param shape: The shape of the output tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function scatterND_<R extends Rank>(
    indices: Tensor|TensorLike, updates: Tensor|TensorLike,
    shape: ShapeMap[R]): Tensor<R> {
  const $indices = convertToTensor(indices, 'indices', 'scatterND', 'int32');
  const $updates = convertToTensor(updates, 'updates', 'scatterND');
  scatter_nd_util.validateInput($updates, $indices, shape);

  const inputs: ScatterNdInputs = {indices: $indices, updates: $updates};
  const attrs: ScatterNdAttrs = {shape};

  // tslint:disable-next-line: no-unnecessary-type-assertion
  return ENGINE.runKernel(
             ScatterNd, inputs as {} as NamedTensorMap,
             attrs as {} as NamedAttrMap) as Tensor<R>;
}