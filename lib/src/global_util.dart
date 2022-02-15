import 'package:tensorflow_wasm/src/engine.dart' show Engine;

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// Note that the identifier globalNameSpace is scoped to this module, but will
// always resolve to the same global object regardless of how the module is
// resolved.
// tslint:disable-next-line:no-any
final _globalNameSpace = GlobalNamespace();

class GlobalNamespace {
  Map<String, Object?>? _tfGlobals;
  Engine? tfengine;
}

// tslint:disable-next-line:no-any
GlobalNamespace getGlobalNamespace() {
  if (_globalNameSpace == null) {
    // TODO:
    // tslint:disable-next-line:no-any
    // let ns: any;
    // if (typeof (window) != 'undefined') {
    //   ns = window;
    // } else if (typeof (global) != 'undefined') {
    //   ns = global;
    // } else if (typeof (process) != 'undefined') {
    //   ns = process;
    // } else if (typeof (self) != 'undefined') {
    //   ns = self;
    // } else {
    //   throw Exception('Could not find a global object');
    // }
    // globalNameSpace = ns;
  }
  return _globalNameSpace;
}

// tslint:disable-next-line:no-any
Map<String, Object?> getGlobalMap() {
  final ns = getGlobalNamespace();
  ns._tfGlobals ??= {};
  return ns._tfGlobals!;
}

/**
 * Returns a globally accessible 'singleton' object.
 *
 * @param key the name of the object
 * @param init a function to initialize to initialize this object
 *             the first time it is fetched.
 */
T getGlobal<T>(String key, T Function() init) {
  final globalMap = getGlobalMap();
  if (globalMap.containsKey(key)) {
    return globalMap[key] as T;
  } else {
    final singleton = init();
    globalMap[key] = singleton;
    return singleton;
  }
}
