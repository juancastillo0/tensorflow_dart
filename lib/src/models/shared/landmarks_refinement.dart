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

// import {Keypoint} from './interfaces/common_interfaces';
// import {LandmarksRefinementConfig} from './interfaces/config_interfaces';

import 'interfaces/common_interfaces.dart';
import 'interfaces/config_interfaces.dart';

import 'dart:math' as Math;

int _getNumberOfRefinedLandmarks(List<LandmarksRefinementConfig> refinements) {
  // Gather all used indexes.
  final indices =
      refinements.expand((refinement) => refinement.indexesMapping).toList();

  if (indices.isEmpty) {
    throw Exception('There should be at least one landmark in indexes mapping');
  }

  int minIndex = indices[0], maxIndex = indices[0];

  final uniqueIndices = indices.toSet();

  uniqueIndices.forEach((index) {
    minIndex = Math.min(minIndex, index);
    maxIndex = Math.max(maxIndex, index);
  });

  // Check that indxes start with 0 and there is no gaps between min and max
  // indexes.
  final numIndices = uniqueIndices.length;

  if (minIndex != 0) {
    throw Exception(
        'Indexes are expected to start with 0 instead of ${minIndex}');
  }

  if (maxIndex + 1 != numIndices) {
    throw Exception(
        'Indexes should have no gaps but ${maxIndex - numIndices + 1} indexes are missing');
  }

  return numIndices;
}

void _refineXY(
  List<int> indexesMapping,
  List<Keypoint> landmarks,
  List<Keypoint> refinedLandmarks,
) {
  for (int i = 0; i < landmarks.length; ++i) {
    final landmark = landmarks[i];
    final refinedLandmark = Keypoint(x: landmark.x, y: landmark.y);
    refinedLandmarks[indexesMapping[i]] = refinedLandmark;
  }
}

double _getZAverage(List<Keypoint> landmarks, List<int> indexes) {
  double zSum = 0;
  for (int i = 0; i < indexes.length; ++i) {
    zSum += landmarks[indexes[i]].z!;
  }
  return zSum / indexes.length;
}

void _refineZ(
  List<int> indexesMapping,
  LandmarksRefinementConfigZRefinement zRefinement,
  List<Keypoint> landmarks,
  List<Keypoint> refinedLandmarks,
) {
  if (zRefinement is String) {
    switch (zRefinement) {
      case 'copy':
        {
          for (int i = 0; i < landmarks.length; ++i) {
            refinedLandmarks[indexesMapping[i]].z = landmarks[i].z;
          }
          break;
        }
      case 'none':
      default:
        {
          // Do nothing and keep Z that is already in refined landmarks.
          break;
        }
    }
  } else {
    final zAverage = _getZAverage(refinedLandmarks, zRefinement as List<int>);
    for (int i = 0; i < indexesMapping.length; ++i) {
      refinedLandmarks[indexesMapping[i]].z = zAverage;
    }
  }
}

/**
 * Refine one set of landmarks with another.
 *
 * @param allLandmarks List of landmarks to use for refinement. They will be
 *     applied to the output in the provided order. Each list should be non
 *     empty and contain the same amount of landmarks as indexes in mapping.
 * @param refinements Refinement instructions for input landmarks.
 *
 * @returns A list of refined landmarks.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_refinement_calculator.cc
List<Keypoint> landmarksRefinement(List<List<Keypoint>> allLandmarks,
    List<LandmarksRefinementConfig> refinements) {
  // Initialize refined landmarks list.
  final numRefinedLandmarks = _getNumberOfRefinedLandmarks(refinements);
  final List<Keypoint> refinedLandmarks = new Array(numRefinedLandmarks);

  // Apply input landmarks to output refined landmarks in provided order.
  for (int i = 0; i < allLandmarks.length; ++i) {
    final landmarks = allLandmarks[i];
    final refinement = refinements[i];

    if (landmarks.length != refinement.indexesMapping.length) {
      // Check number of landmarks in mapping and stream are the same.
      throw Exception(
          'There are ${landmarks.length} refinement landmarks while mapping has ${refinement.indexesMapping.length}');
    }

    // Refine X and Y.
    _refineXY(refinement.indexesMapping, landmarks, refinedLandmarks);

    // Refine Z.
    _refineZ(refinement.indexesMapping, refinement.zRefinement, landmarks,
        refinedLandmarks);

    // Visibility and presence are not currently refined and are left as `0`.
  }

  return refinedLandmarks;
}
