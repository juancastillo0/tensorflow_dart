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

import 'dart:async';

import 'package:http/http.dart' as http;
import 'package:tensorflow_wasm/src/util_base.dart' as util;

// import {Platform} from './platforms/platform';
// import {isPromise} from './util_base';

// Expects flags from URL in the format ?tfjsflags=FLAG1:1,FLAG2:true.
const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
const kIsWeb = identical(0, 0.0);

typedef FlagValue = Object; // number|boolean;
typedef FlagEvaluationFn = FutureOr<FlagValue> Function();
typedef Flags = Map<String, FlagValue>;

class FlagRegistryEntry {
  final FlagEvaluationFn evaluationFn;
  final void Function(FlagValue value)? setHook;

  FlagRegistryEntry({
    required this.evaluationFn,
    this.setHook,
  });
}

typedef RequestInit = http.BaseRequest;

typedef FetchFn = Future<http.StreamedResponse> Function(
  Uri,
  RequestInit?,
);

class TFPlatform {
  final FetchFn fetch;

  Future<http.Response> fetchAndParse(
    Uri uri, [
    RequestInit? req,
  ]) async {
    final stream = await fetch(uri, req);
    final response = await http.Response.fromStream(stream);
    return response;
  }

  TFPlatform({
    required this.fetch,
  });
}

/**
 * The environment contains evaluated flags as well as the registered platform.
 * This is always used as a global singleton and can be retrieved with
 * `tf.env()`.
 *
 * @doc {heading: 'Environment'}
 */
class Environment {
  Flags flags = {};
  final flagRegistry = <String, FlagRegistryEntry>{};

  Flags urlFlags = {};

  String? platformName;
  TFPlatform? platform;
  final Object global;
  // Jasmine spies on this in 'environment_test.ts'
  // getQueryParams = getQueryParams;

  // tslint:disable-next-line: no-any
  Environment(this.global) {
    // this.populateURLFlags(); // TODO:

    setPlatform(
      kIsWeb ? 'WEB' : 'NATIVE',
      TFPlatform(
        fetch: (uri, base) {
          if (base is http.MultipartRequest) return base.send();
          if (base is http.Request) {
            final req = http.Request(base.method, uri);
            if (base.method != 'GET' && base.method != 'HEAD') {
              req.bodyBytes = base.bodyBytes;
            }
            req.headers.addAll(req.headers);
            // req.contentLength = req.contentLength;
            req.persistentConnection = req.persistentConnection;
            req.encoding = req.encoding;
            req.followRedirects = req.followRedirects;
            req.maxRedirects = req.maxRedirects;
            return req.send();
          }
          return http.Request('GET', uri).send();
        },
      ),
    );
  }

  setPlatform(String platformName, TFPlatform platform) {
    if (this.platform != null) {
      if (!(env().getBool('IS_TEST') || env().getBool('PROD'))) {
        util.log.warning(
            'Platform ${this.platformName} has already been set. ' +
                'Overwriting the platform with ${platformName}.');
      }
    }
    this.platformName = platformName;
    this.platform = platform;
  }

  registerFlag(
    String flagName,
    FlagEvaluationFn evaluationFn, [
    void Function(FlagValue)? setHook,
  ]) {
    this.flagRegistry[flagName] =
        FlagRegistryEntry(evaluationFn: evaluationFn, setHook: setHook);

    // Override the flag value from the URL. This has to happen here because
    // the environment is initialized before flags get registered.
    if (this.urlFlags[flagName] != null) {
      final flagValue = this.urlFlags[flagName]!;
      if (!(env().getBool('IS_TEST') || env().getBool('PROD'))) {
        util.log.warning(
            'Setting feature override from URL ${flagName}: ${flagValue}.');
      }
      this.set(flagName, flagValue);
    }
  }

  Future<FlagValue> getAsync(String flagName) async {
    if (this.flags.containsKey(flagName)) {
      return this.flags[flagName]!;
    }

    this.flags[flagName] = await this._evaluateFlag(flagName);
    return this.flags[flagName]!;
  }

  FlagValue get(String flagName) {
    if (this.flags.containsKey(flagName)) {
      return this.flags[flagName]!;
    }

    final flagValue = this._evaluateFlag(flagName);
    if (flagValue is Future) {
      throw Exception('Flag ${flagName} cannot be synchronously evaluated. ' +
          'Please use getAsync() instead.');
    }

    this.flags[flagName] = flagValue;
    return this.flags[flagName]!;
  }

  num getNumber(String flagName) {
    return this.get(flagName) as num;
  }

  bool getBool(String flagName) {
    return this.get(flagName) as bool;
  }

  Flags getFlags() {
    return this.flags;
  }

  // For backwards compatibility.
  Flags get features {
    return this.flags;
  }

  void set(String flagName, FlagValue value) {
    if (this.flagRegistry[flagName] == null) {
      throw Exception(
          'Cannot set flag ${flagName} as it has not been registered.');
    }
    this.flags[flagName] = value;
    if (this.flagRegistry[flagName]!.setHook != null) {
      this.flagRegistry[flagName]!.setHook!(value);
    }
  }

  FutureOr<FlagValue> _evaluateFlag(String flagName) {
    if (this.flagRegistry[flagName] == null) {
      throw Exception(
          "Cannot evaluate flag '${flagName}': no evaluation function found.");
    }
    return this.flagRegistry[flagName]!.evaluationFn();
  }

  void setFlags(Flags flags) {
    this.flags = {...flags};
  }

  reset() {
    this.flags = {};
    this.urlFlags = {};
    // TODO:
    // this.populateURLFlags();
  }

  // private populateURLFlags(): void {
  //   if (typeof this.global === 'undefined' ||
  //       typeof this.global.location === 'undefined' ||
  //       typeof this.global.location.search === 'undefined') {
  //     return;
  //   }

  //   final urlParams = this.getQueryParams(this.global.location.search);
  //   if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
  //     final keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
  //     keyValues.forEach((keyValue) => {
  //       final [key, value] = keyValue.split(':') as [string, string];
  //       this.urlFlags[key] = parseValue(key, value);
  //     });
  //   }
  // }
}

// export function getQueryParams(queryString: string): {[key: string]: string} {
//   final params = {};
//   queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => {
//     decodeParam(params, t[0], t[1]);
//     return t.join('=');
//   });
//   return params;
// }

// function decodeParam(
//     params: {[key: string]: string}, name: string, value?: string) {
//   params[decodeURIComponent(name)] = decodeURIComponent(value || '');
// }

FlagValue parseValue(String flagName, String value) {
  value = value.toLowerCase();
  if (value == 'true' || value == 'false') {
    return value == 'true';
  } else if ('${value}' == value) {
    return value;
  }
  throw Exception(
      'Could not parse value flag value ${value} for flag ${flagName}.');
}

/**
 * Returns the current environment (a global singleton).
 *
 * The environment object contains the evaluated feature values as well as the
 * active platform.
 *
 * @doc {heading: 'Environment'}
 */
Environment env() {
  return ENV!;
}

Environment? ENV = null;
void setEnvironmentGlobal(Environment environment) {
  ENV = environment;
}
