/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// import {ImageSize, Keypoint} from './interfaces/common_interfaces';

import 'interfaces/common_interfaces.dart';

List<Keypoint> normalizedKeypointsToKeypoints(
  List<Keypoint> normalizedKeypoints,
  ImageSize imageSize,
) {
  return normalizedKeypoints.map((normalizedKeypoint) {
    return normalizedKeypoint.copyWith(
      x: normalizedKeypoint.x * imageSize.width,
      y: normalizedKeypoint.y * imageSize.height,
      z: normalizedKeypoint.z == null
          ? null
          : Nullable(normalizedKeypoint.z! * imageSize.width),
    );
  }).toList();
}
