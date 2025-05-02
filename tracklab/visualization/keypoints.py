from tracklab.visualization import DetectionVisualizer
from tracklab.utils.cv2 import draw_keypoints

class DefaultKeypoints(DetectionVisualizer):
    def __init__(self, threshold=0.4, print_confidence=False):
        self.threshold = threshold
        self.print_confidence = print_confidence
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_gt is not None:
            color_kp = self.color(detection_gt, is_prediction=False)
            if color_kp:
                draw_keypoints(
                    detection_gt,
                    image,
                    color_kp,
                    threshold=self.threshold,
                )
        if detection_pred is not None:
            color_kp = self.color(detection_pred, is_prediction=True)
            if color_kp:
                draw_keypoints(
                    detection_pred,
                    image,
                    color_kp,
                    threshold=self.threshold,
                    print_confidence=self.print_confidence,
                )

class FullKeypoints(DefaultKeypoints):
    def __init__(self):
        super().__init__(threshold=0., print_confidence=True)
