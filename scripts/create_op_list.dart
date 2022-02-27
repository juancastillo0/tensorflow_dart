import 'dart:convert';
import 'dart:io';

void main(List<String> args) async {
  final rootDir = [
    '/',
    ...Platform.script.pathSegments
        .sublist(0, Platform.script.pathSegments.length - 2)
  ];

  final opListDir = Directory(
    [...rootDir, 'scripts', 'op_list'].join(Platform.pathSeparator),
  );

  final files = opListDir.listSync().cast<File>();

  final output = await Future.wait(files.map((file) async {
    final str = await file.readAsString();
    final List<Map<String, Object?>> json = (jsonDecode(str) as List).cast();
    final fileName = file.uri.pathSegments.last;

    return MapEntry(fileName.substring(0, fileName.length - 5), '''
import '../types.dart';

const opMappers = [
  ${json.map((e) {
      final _inputs = (e['inputs'] as List? ?? []).map((e) {
        final mapper = _makeMapper(e);
        return 'InputParamMapper(start:${e['start']},'
            ' end:${e['end']}, mapper: $mapper,),';
      }).join();
      final _attrs = (e['attrs'] as List? ?? []).map((e) {
        final mapper = _makeMapper(e);
        return 'AttrParamMapper(tfName:${_dartString(e['tfName'])},'
            ' tfDeprecatedName:${_dartString(e['tfDeprecatedName'])}, mapper: $mapper,),';
      }).join();

      return '''OpMapper(
      tfOpName: ${_dartString(e['tfOpName'])},
      category: ${_dartString(e['category'])},
      inputs: [$_inputs],
      attrs: [$_attrs],
      ),''';
    }).join()}
];
''');
  }));

  final opListOutDir = await Directory([
    ...rootDir,
    'lib',
    'src',
    'converter',
    'operations',
    'op_list'
  ].join(Platform.pathSeparator))
      .create();

  await Future.wait(output.map((e) async {
    final file = await File(
      [...opListOutDir.uri.pathSegments, '${e.key}.dart']
          .join(Platform.pathSeparator),
    ).create();

    return file.writeAsString(e.value);
  }));

  final result = await Process.run(
    'fvm',
    [
      'dart',
      'format',
      'lib/src/converter/operations/op_list'
          .replaceAll('/', Platform.pathSeparator),
    ],
  );
  if (result.stderr != null) {
    print(result.stderr);
  }
}

String _dartString(Object? obj) {
  if (obj is! String?) {
    throw Exception(
      '_dartString: $obj ${obj.runtimeType}, expected String?.',
    );
  }
  return obj == null ? 'null' : "'$obj'";
}

String _makeMapper(Map<String, Object?> json) {
  return '''
ParamMapper(
name: ${_dartString(json['name'])},
type: ${_dartString(json['type'])},
defaultValue: ${json['defaultValue'] is String ? _dartString(json['defaultValue']) : json['defaultValue']},
notSupported: ${json['notSupported']},
)
''';
}
