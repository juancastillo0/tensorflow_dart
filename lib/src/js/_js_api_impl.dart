import 'package:js/js.dart';
import 'package:universal_html/html.dart';
import 'package:universal_html/js_util.dart';

@JS()
@anonymous
class CreateImageBitmapOptions {
  external factory CreateImageBitmapOptions({
    String? imageOrientation,
    String? premultiplyAlpha,
    String? colorSpaceConversion,
    int? resizeWidth,
    int? resizeHeight,
    String? resizeQuality,
  });
}

Future<Object> createImageBitmap(
  Object image, [
  CreateImageBitmapOptions? options,
]) {
  return promiseToFuture(
    callMethod(
      window,
      'createImageBitmap',
      [image, if (options != null) options],
    ),
  );
}

bool isImageBitmapFullySupported() => hasProperty(window, 'createImageBitmap');
