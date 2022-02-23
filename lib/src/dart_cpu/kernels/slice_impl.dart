export function sliceImpl(
    vals: BackendValues, begin: number[], size: number[], shape: number[],
    dtype: DataType): BackendValues {
  const isContinous = slice_util.isSliceContinous(shape, begin, size);
  const length = util.sizeFromShape(size);
  const xStrides = util.computeStrides(shape);

  if (isContinous) {
    const flatOffset = slice_util.computeFlatOffset(begin, xStrides);

    if (dtype === 'string') {
      return (vals as Uint8Array[]).slice(flatOffset, flatOffset + length);
    }

    return (vals as TypedArray).subarray(flatOffset, flatOffset + length);
  }

  const decodedData = dtype === 'string' ?
      backend_util.fromUint8ToStringArray(vals as Uint8Array[]) :
      vals as TypedArray;

  const inBuf = buffer(shape, dtype, decodedData);
  const outBuf = buffer(size, dtype);
  for (let i = 0; i < outBuf.size; ++i) {
    const outLoc = outBuf.indexToLoc(i);
    const inLoc = outLoc.map((idx: number, j) => idx + begin[j]);
    outBuf.set(inBuf.get(...inLoc), ...outLoc);
  }

  if (dtype === 'string') {
    return backend_util.fromStringArrayToUint8(outBuf.values as string[]);
  }
  return outBuf.values as TypedArray;
}