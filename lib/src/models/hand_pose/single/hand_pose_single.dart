import '../../hand_pose_single/hand_pose_single.dart' as single;
import '../hand_pose.dart';

/**
 * Loads the MediaPipeHands model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeHands loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeHandsTfjsModelConfig`
 * interface.
 */
Future<HandDetector> load(MediaPipeHandsTfjsModelConfig modelConfig) async {
  final model = await single.load();
  return _SingleHandDetector(model);
}

class _SingleHandDetector implements HandDetector {
  final single.HandPose handpose;

  _SingleHandDetector(this.handpose);

  @override
  void dispose() {}

  @override
  void reset() {}

  @override
  Future<List<Hand>> estimateHands(
    HandDetectorInput input,
    MediaPipeHandsTfjsEstimationConfig? estimationConfig,
  ) async {
    final est = await handpose.estimateHands(
      input,
      flipHorizontal: estimationConfig?.flipHorizontal ?? false,
    );

    print('estimateHands $est');

    return [
      ...est.map(
        (e) {
          print(e.handInViewConfidence);
          return Hand(
            handedness: Handedness.left,
            keypoints:
                e.landmarks.map((e) => Keypoint(x: e[0], y: e[1])).toList(),
            score: e.handInViewConfidence,
          );
        },
      )
    ];
  }
}
