/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

/**
 * Inserts a value into a sorted array. This method allows duplicate, meaning it
 * allows inserting duplicate value, in which case, the element will be inserted
 * at the lowest index of the value.
 * @param arr The array to modify.
 * @param element The element to insert.
 * @param comparator Optional. If no comparator is specified, elements are
 * compared using array_util.defaultComparator, which is suitable for Strings
 * and Numbers in ascending arrays. If the array contains multiple instances of
 * the target value, the left-most instance will be returned. To provide a
 * comparator, it should take 2 arguments to compare and return a negative,
 * zero, or a positive number.
 */
void binaryInsert<T>(
  List<T> arr,
  T element, [
  int Function(T a, T b)? comparator,
]) {
  final index = binarySearch(arr, element, comparator);
  final insertionPoint = index < 0 ? -(index + 1) : index;
  arr.insert(insertionPoint, element);
}

/**
 * Searches the array for the target using binary search, returns the index
 * of the found element, or position to insert if element not found. If no
 * comparator is specified, elements are compared using array_
 * util.defaultComparator, which is suitable for Strings and Numbers in
 * ascending arrays. If the array contains multiple instances of the target
 * value, the left-most instance will be returned.
 * @param arr The array to be searched in.
 * @param target The target to be searched for.
 * @param comparator Should take 2 arguments to compare and return a negative,
 *    zero, or a positive number.
 * @return Lowest index of the target value if found, otherwise the insertion
 *    point where the target should be inserted, in the form of
 *    (-insertionPoint - 1).
 */
int binarySearch<T>(
  List<T> arr,
  T target, [
  int Function(T a, T b)? comparator,
]) {
  return _binarySearch(
    arr,
    target,
    comparator ?? Comparable.compare as dynamic,
  );
}

/**
 * Compares its two arguments for order.
 * @param a The first element to be compared.
 * @param b The second element to be compared.
 * @return A negative number, zero, or a positive number as the first
 *     argument is less than, equal to, or greater than the second.
 */
// int _defaultComparator<T>(T a, T b) {
//   return a > b
//       ? 1
//       : a < b
//           ? -1
//           : 0;
// }

int _binarySearch<T>(
  List<T> arr,
  T target,
  int Function(T a, T b) comparator,
) {
  int left = 0;
  int right = arr.length;
  int middle = 0;
  var found = false;
  while (left < right) {
    middle = left + ((right - left) >>> 1);
    final compareResult = comparator(target, arr[middle]);
    if (compareResult > 0) {
      left = middle + 1;
    } else {
      right = middle;
      // If compareResult is 0, the value is found. We record it is found,
      // and then keep looking because there may be duplicate.
      found = compareResult == 0;
    }
  }

  return found ? left : -left - 1;
}
