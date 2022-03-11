class CreateImageBitmapOptions {
  final String? imageOrientation;
  final String? premultiplyAlpha;
  final String? colorSpaceConversion;
  final int? resizeWidth;
  final int? resizeHeight;
  final String? resizeQuality;

  CreateImageBitmapOptions({
    this.imageOrientation,
    this.premultiplyAlpha,
    this.colorSpaceConversion,
    this.resizeWidth,
    this.resizeHeight,
    this.resizeQuality,
  });
}

Future<Object> createImageBitmap(
  Object image, [
  CreateImageBitmapOptions? options,
]) {
  throw UnsupportedError('Only available in the browser');
}

bool isImageBitmapFullySupported() => false;
