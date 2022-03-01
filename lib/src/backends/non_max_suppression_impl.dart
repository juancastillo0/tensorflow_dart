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

// import {TypedArray} from '../types';
// import {binaryInsert} from './non_max_suppression_util';

import 'dart:math' as Math;
import 'package:collection/collection.dart';

import 'non_max_suppression_util.dart';

/**
 * Implementation of the NonMaxSuppression kernel shared between webgl and cpu.
 */
class _Candidate {
  double score;
  final int boxIndex;
  int suppressBeginIndex;

  _Candidate({
    required this.score,
    required this.boxIndex,
    required this.suppressBeginIndex,
  });
}

class NonMaxSuppressionResult {
  final List<int> selectedIndices;
  final List<double>? selectedScores;
  final int? validOutputs;

  NonMaxSuppressionResult({
    required this.selectedIndices,
    this.selectedScores,
    this.validOutputs,
  });
}

NonMaxSuppressionResult nonMaxSuppressionV3Impl(
  List<int> boxes,
  List<double> scores,
  int maxOutputSize,
  double iouThreshold,
  double scoreThreshold,
) {
  return _nonMaxSuppressionImpl(boxes, scores, maxOutputSize, iouThreshold,
      scoreThreshold, 0 /* softNmsSigma */);
}

NonMaxSuppressionResult nonMaxSuppressionV4Impl(
  List<int> boxes,
  List<double> scores,
  int maxOutputSize,
  double iouThreshold,
  double scoreThreshold,
  bool padToMaxOutputSize,
) {
  return _nonMaxSuppressionImpl(boxes, scores, maxOutputSize, iouThreshold,
      scoreThreshold, 0 /* softNmsSigma */,
      returnScoresTensor: false,
      padToMaxOutputSize: padToMaxOutputSize,
      returnValidOutputs: true);
}

NonMaxSuppressionResult nonMaxSuppressionV5Impl(
  List<int> boxes,
  List<double> scores,
  int maxOutputSize,
  double iouThreshold,
  double scoreThreshold,
  double softNmsSigma,
) {
  return _nonMaxSuppressionImpl(
      boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma,
      returnScoresTensor: true /* returnScoresTensor */);
}

NonMaxSuppressionResult _nonMaxSuppressionImpl(
  List<int> boxes,
  List<double> scores,
  int maxOutputSize,
  double iouThreshold,
  double scoreThreshold,
  double softNmsSigma, {
  bool returnScoresTensor = false,
  bool padToMaxOutputSize = false,
  bool returnValidOutputs = false,
}) {
  // The list is sorted in ascending order, so that we can always pop the
  // candidate with the largest score in O(1) time.
  final candidates = <_Candidate>[];

  for (int i = 0; i < scores.length; i++) {
    if (scores[i] > scoreThreshold) {
      candidates.add(
          _Candidate(score: scores[i], boxIndex: i, suppressBeginIndex: 0));
    }
  }

  candidates.sort(_ascendingComparator);

  // If softNmsSigma is 0, the outcome of this algorithm is exactly same as
  // before.
  final scale = softNmsSigma > 0 ? (-0.5 / softNmsSigma) : 0.0;

  final List<int> selectedIndices = [];
  final List<double> selectedScores = [];

  while (selectedIndices.length < maxOutputSize && candidates.length > 0) {
    final candidate = candidates.removeLast();
    final originalScore = candidate.score;

    if (originalScore < scoreThreshold) {
      break;
    }

    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if candidate's score should be suppressed. We use
    // suppressBeginIndex to track and ensure a candidate can be suppressed
    // by a selected box no more than once. Also, if the overlap exceeds
    // iouThreshold, we simply ignore the candidate.
    bool ignoreCandidate = false;
    for (int j = selectedIndices.length - 1;
        j >= candidate.suppressBeginIndex;
        --j) {
      final iou =
          intersectionOverUnion(boxes, candidate.boxIndex, selectedIndices[j]);

      if (iou >= iouThreshold) {
        ignoreCandidate = true;
        break;
      }

      candidate.score =
          candidate.score * _suppressWeight(iouThreshold, scale, iou);

      if (candidate.score <= scoreThreshold) {
        break;
      }
    }

    // At this point, if `candidate.score` has not dropped below
    // `scoreThreshold`, then we know that we went through all of the
    // previous selections and can safely update `suppressBeginIndex` to the
    // end of the selected array. Then we can re-insert the candidate with
    // the updated score and suppressBeginIndex back in the candidate list.
    // If on the other hand, `candidate.score` has dropped below the score
    // threshold, we will not add it back to the candidates list.
    candidate.suppressBeginIndex = selectedIndices.length;

    if (!ignoreCandidate) {
      // Candidate has passed all the tests, and is not suppressed, so
      // select the candidate.
      if (candidate.score == originalScore) {
        selectedIndices.add(candidate.boxIndex);
        selectedScores.add(candidate.score);
      } else if (candidate.score > scoreThreshold) {
        // Candidate's score is suppressed but is still high enough to be
        // considered, so add back to the candidates list.
        binaryInsert(candidates, candidate, _ascendingComparator);
      }
    }
  }

  // NonMaxSuppressionV4 feature: padding output to maxOutputSize.
  final validOutputs = selectedIndices.length;
  final elemsToPad = maxOutputSize - validOutputs;

  if (padToMaxOutputSize && elemsToPad > 0) {
    selectedIndices.addAll(List.filled(elemsToPad, 0));
    selectedScores.addAll(List.filled(elemsToPad, 0.0));
  }

  final result = NonMaxSuppressionResult(
    selectedIndices: selectedIndices,
    selectedScores: returnScoresTensor ? selectedScores : null,
    validOutputs: returnValidOutputs ? validOutputs : null,
  );

  return result;
}

double intersectionOverUnion(List<int> boxes, int i, int j) {
  final iCoord = boxes.slice(i * 4, i * 4 + 4);
  final jCoord = boxes.slice(j * 4, j * 4 + 4);
  final yminI = Math.min(iCoord[0], iCoord[2]);
  final xminI = Math.min(iCoord[1], iCoord[3]);
  final ymaxI = Math.max(iCoord[0], iCoord[2]);
  final xmaxI = Math.max(iCoord[1], iCoord[3]);
  final yminJ = Math.min(jCoord[0], jCoord[2]);
  final xminJ = Math.min(jCoord[1], jCoord[3]);
  final ymaxJ = Math.max(jCoord[0], jCoord[2]);
  final xmaxJ = Math.max(jCoord[1], jCoord[3]);
  final areaI = (ymaxI - yminI) * (xmaxI - xminI);
  final areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
  if (areaI <= 0 || areaJ <= 0) {
    return 0.0;
  }
  final intersectionYmin = Math.max(yminI, yminJ);
  final intersectionXmin = Math.max(xminI, xminJ);
  final intersectionYmax = Math.min(ymaxI, ymaxJ);
  final intersectionXmax = Math.min(xmaxI, xmaxJ);
  final intersectionArea = Math.max(intersectionYmax - intersectionYmin, 0.0) *
      Math.max(intersectionXmax - intersectionXmin, 0.0);
  return intersectionArea / (areaI + areaJ - intersectionArea);
}

// A Gaussian penalty function, this method always returns values in [0, 1].
// The weight is a function of similarity, the more overlap two boxes are, the
// smaller the weight is, meaning highly overlapping boxe will be significantly
// penalized. On the other hand, a non-overlapping box will not be penalized.
double _suppressWeight(double iouThreshold, double scale, double iou) {
  final weight = Math.exp(scale * iou * iou);
  return iou <= iouThreshold ? weight : 0.0;
}

int _ascendingComparator(_Candidate c1, _Candidate c2) {
  // For objects with same scores, we make the object with the larger index go
  // first. In an array that pops from the end, this means that the object with
  // the smaller index will be popped first. This ensures the same output as
  // the TensorFlow python version.
  return (c1.score == c2.score)
      ? (c2.boxIndex - c1.boxIndex)
      : (c1.score - c2.score).toInt();
}
