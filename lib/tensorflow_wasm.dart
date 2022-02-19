/// Support for doing something awesome.
///
/// More dartdocs go here.
library tensorflow_wasm;

export 'src/base.dart';
export 'src/wasm.dart';

extension MapGetSet<K, V> on Map<K, V> {
  V? get(K key) => this[key];
  V? set(K key, V value) {
    final prev = this[key];
    this[key] = value;
    return prev;
  }
}

extension SliceList<T> on List<T> {
  List<T> slice(int begin, [int? end]) {
    return sublist(
      begin < 0 ? length + begin : begin,
      end == null
          ? length
          : end < 0
              ? length + end
              : end,
    );
  }
}

// TODO: Export any libraries intended for clients of this package.
