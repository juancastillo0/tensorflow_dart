import 'types.dart';

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

// import {IOHandler, LoadOptions} from './types';

typedef IORouter = IOHandler? Function(
  // : string|string[]
  List<String> url,
  LoadOptions? loadOptions,
);

class IORouterRegistry {
  // Singleton instance.
  static IORouterRegistry? instance;

  final List<IORouter> saveRouters = [];
  final List<IORouter> loadRouters = [];

  IORouterRegistry._();

  static IORouterRegistry getInstance() {
    return IORouterRegistry.instance ??= IORouterRegistry._();
  }

  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  static void registerSaveRouter(IORouter saveRouter) {
    IORouterRegistry.getInstance().saveRouters.add(saveRouter);
  }

  /**
   * Register a load-handler router.
   *
   * @param loadRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `load` method defined or `null`.
   */
  static void registerLoadRouter(IORouter loadRouter) {
    IORouterRegistry.getInstance().loadRouters.add(loadRouter);
  }

  /**
   * Look up IOHandler for saving, given a URL-like string.
   *
   * @param url
   * @returns If only one match is found, an instance of IOHandler with the
   * `save` method defined. If no match is found, `null`.
   * @throws Error, if more than one match is found.
   */
  static List<IOHandler> getSaveHandlers(List<String> url) {
    return IORouterRegistry._getHandlers(url, 'save');
  }

  /**
   * Look up IOHandler for loading, given a URL-like string.
   *
   * @param url
   * @param loadOptions Optional, custom load options.
   * @returns All valid handlers for `url`, given the currently registered
   *   handler routers.
   */
  static List<IOHandler> getLoadHandlers(
    // : string|string[]
    List<String> url, [
    LoadOptions? loadOptions,
  ]) {
    return IORouterRegistry._getHandlers(url, 'load', loadOptions);
  }

  static List<IOHandler> _getHandlers(
    // : string|string[]
    List<String> url,
    // : 'save'|'load'
    String handlerType, [
    LoadOptions? loadOptions,
  ]) {
    final List<IOHandler> validHandlers = [];
    final routers = handlerType == 'load'
        ? IORouterRegistry.getInstance().loadRouters
        : IORouterRegistry.getInstance().saveRouters;
    routers.forEach((router) {
      final handler = router(url, loadOptions);
      if (handler != null) {
        validHandlers.add(handler);
      }
    });
    return validHandlers;
  }
}

void registerSaveRouter(IORouter loudRouter) =>
    IORouterRegistry.registerSaveRouter(loudRouter);
void registerLoadRouter(IORouter loudRouter) =>
    IORouterRegistry.registerLoadRouter(loudRouter);

List<IOHandler> getSaveHandlers(
  // string|string[]
  List<String> url,
) =>
    IORouterRegistry.getSaveHandlers(url);

List<IOHandler> getLoadHandlers(
  // string|string[]
  List<String> url, [
  LoadOptions? loadOptions,
]) =>
    IORouterRegistry.getLoadHandlers(url, loadOptions);
