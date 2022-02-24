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

// import * as seedrandom from 'seedrandom';

// import {expectNumbersClose, testEpsilon} from '../test_util';
// import {TypedArray} from '../types';

import 'dart:math' as Math;

abstract class RandomBase {
  double nextValue();
}

Math.Random _makeRandom(int? seed) {
  final seedValue = seed ?? Math.Random().nextInt((1e10).toInt());
  return Math.Random(seedValue);
}

// export interface RandomGamma {
//   nextValue(): number;
// }

// export interface RandNormalDataTypes {
//   float32: Float32Array;
//   int32: Int32Array;
// }

// export interface RandGammaDataTypes {
//   float32: Float32Array;
//   int32: Int32Array;
// }

// https://en.wikipedia.org/wiki/Marsaglia_polar_method
class MPRandGauss implements RandomBase {
  final double mean;
  final double stdDev;
  double nextVal = double.nan;
  final bool truncated;
  late final double? upper;
  late final double? lower;
  late final Math.Random random;

  MPRandGauss({
    required this.mean,
    required this.stdDev,
    this.truncated = false,
    int? seed,
  }) : random = _makeRandom(seed) {
    if (this.truncated) {
      this.upper = this.mean + this.stdDev * 2;
      this.lower = this.mean - this.stdDev * 2;
    }
  }

  /** Returns next sample from a Gaussian distribution. */
  double nextValue() {
    if (!this.nextVal.isNaN) {
      final value = this.nextVal;
      this.nextVal = double.nan;
      return value;
    }

    late double resultX, resultY;
    bool isValid = false;
    while (!isValid) {
      double v1, v2, s;
      do {
        v1 = 2 * this.random.nextDouble() - 1;
        v2 = 2 * this.random.nextDouble() - 1;
        s = v1 * v1 + v2 * v2;
      } while (s >= 1 || s == 0);

      final mul = Math.sqrt(-2.0 * Math.log(s) / s);
      resultX = this.mean + this.stdDev * v1 * mul;
      resultY = this.mean + this.stdDev * v2 * mul;

      if (!this.truncated || this._isValidTruncated(resultX)) {
        isValid = true;
      }
    }

    if (!this.truncated || this._isValidTruncated(resultY)) {
      this.nextVal = this._convertValue(resultY);
    }
    return this._convertValue(resultX);
  }

  /** Handles proper rounding for non-floating-point numbers. */
  double _convertValue(double value) {
    // if (this.dtype == null || this.dtype == 'float32') {
    return value;
    // }
    // return (value).round();
  }

  /** Returns true if less than 2-standard-deviations from the mean. */
  bool _isValidTruncated(double value) {
    return value <= this.upper! && value >= this.lower!;
  }
}

// Marsaglia, George, and Wai Wan Tsang. 2000. "A Simple Method for Generating
// Gamma Variables."
class RandGamma implements RandomBase {
  final double alpha;
  final double beta;
  late final double d;
  late final double c;
  late final Math.Random randu;
  late final MPRandGauss randn;

  RandGamma({
    required this.alpha,
    required this.beta,
    int? seed,
  }) {
    // this.beta = 1 / beta;  // convert rate to scale parameter

    final seedValue = seed ?? Math.Random().nextInt((1e10).toInt());
    this.randu = Math.Random(seedValue);
    // TODO: had dtype
    this.randn = MPRandGauss(
        mean: 0,
        stdDev: 1,
        truncated: false,
        seed: this.randu.nextInt((1e10).toInt()));

    if (alpha < 1) {
      this.d = alpha + (2 / 3);
    } else {
      this.d = alpha - (1 / 3);
    }
    this.c = 1 / Math.sqrt(9 * this.d);
  }

  /** Returns next sample from a gamma distribution. */
  double nextValue() {
    double x2, v0, v1, x, u, v;
    while (true) {
      do {
        x = this.randn.nextValue();
        v = 1.0 + (this.c * x);
      } while (v <= 0);
      v *= v * v;
      x2 = x * x;
      v0 = 1 - (0.331 * x2 * x2);
      v1 = (0.5 * x2) + (this.d * (1 - v + Math.log(v)));
      u = this.randu.nextDouble();
      if (u < v0 || Math.log(u) < v1) {
        break;
      }
    }
    v = this.beta * this.d * v;
    if (this.alpha < 1) {
      v *= Math.pow(this.randu.nextDouble(), 1 / this.alpha);
    }
    return this._convertValue(v);
  }

  /** Handles proper rounding for non-floating-point numbers. */
  double _convertValue(double value) {
    // if (this.dtype == 'float32') {
    return value;
    // }
    // return (value).round();
  }
}

class UniformRandom implements RandomBase {
  final double min;
  final double range;
  final Math.Random random;

  UniformRandom({
    this.min = 0,
    double max = 1,
    int? seed,
  })  : range = max - min,
        this.random = _makeRandom(seed) {
    if (range <= 0) {
      throw Exception(
          'The difference between max=${max} - min=${min} <= 0. max should be greater than min.');
    }
    if (!this._canReturnFloat() && this.range <= 1) {
      throw Exception(
          'The difference between ${max} - ${min} <= 1 and dtype is not float.');
    }
  }

  /** Handles proper rounding for non floating point numbers. */
  bool _canReturnFloat() =>
      true; // (this.dtype == null || this.dtype == 'float32');

  double _convertValue(double value) {
    // if (this._canReturnFloat()) {
    return value;
    // }
    // return (value).round();
  }

  double nextValue() {
    return this._convertValue(this.min + this.range * this.random.nextDouble());
  }
}

void jarqueBeraNormalityTest(List<double> values) {
  // https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test
  final n = values.length;
  final s = _skewness(values);
  final k = _kurtosis(values);
  final jb = n / 6 * (Math.pow(s, 2) + 0.25 * Math.pow(k - 3, 2));
  // JB test requires 2-degress of freedom from Chi-Square @ 0.95:
  // http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
  const CHI_SQUARE_2DEG = 5.991;
  if (jb > CHI_SQUARE_2DEG) {
    throw Exception('Invalid p-value for JB: ${jb}');
  }
}

// void expectArrayInMeanStdRange(List<double> actual, double expectedMean,
//     double expectedStdDev, double? epsilon) {
//   if (epsilon == null) {
//     epsilon = testEpsilon();
//   }
//   final actualMean = _mean(actual);
//   expectNumbersClose(actualMean, expectedMean, epsilon);
//   expectNumbersClose(
//       _standardDeviation(actual, actualMean), expectedStdDev, epsilon);
// }

double _mean(List<double> values) {
  double sum = 0;
  for (int i = 0; i < values.length; i++) {
    sum += values[i];
  }
  return sum / values.length;
}

double _standardDeviation(List<double> values, double mean) {
  double squareDiffSum = 0;
  for (int i = 0; i < values.length; i++) {
    final diff = values[i] - mean;
    squareDiffSum += diff * diff;
  }
  return Math.sqrt(squareDiffSum / values.length);
}

double _kurtosis(List<double> values) {
  // https://en.wikipedia.org/wiki/Kurtosis
  final valuesMean = _mean(values);
  final n = values.length;
  double sum2 = 0;
  double sum4 = 0;
  for (int i = 0; i < n; i++) {
    final v = values[i] - valuesMean;
    sum2 += Math.pow(v, 2);
    sum4 += Math.pow(v, 4);
  }
  return (1 / n) * sum4 / Math.pow((1 / n) * sum2, 2);
}

double _skewness(List<double> values) {
  // https://en.wikipedia.org/wiki/Skewness
  final valuesMean = _mean(values);
  final n = values.length;
  double sum2 = 0;
  double sum3 = 0;
  for (int i = 0; i < n; i++) {
    final v = values[i] - valuesMean;
    sum2 += Math.pow(v, 2);
    sum3 += Math.pow(v, 3);
  }
  return (1 / n) * sum3 / Math.pow((1 / (n - 1)) * sum2, 3 / 2);
}
