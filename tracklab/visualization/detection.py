import cv2

from tracklab.core.visualizer import DetectionVisualizer
from tracklab.utils.cv2 import draw_bbox


class DefaultDetectionVisualizer(DetectionVisualizer):
    def __init__(self, draw_prediction=True, draw_ground_truth=False):
        super().__init__()
        self.draw_prediction = draw_prediction
        self.draw_ground_truth = draw_ground_truth

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if self.draw_ground_truth and detection_gt is not None:
            color_gt = self.color(detection_gt, is_prediction=False, color_type="bbox")
            draw_bbox(detection_gt, image, color_gt, 1, None, None, None, None)
        if self.draw_prediction and detection_pred is not None:
            color_pred = self.color(detection_pred, is_prediction=True, color_type="bbox")
            draw_bbox(detection_pred, image, color_pred, 1, None, None, None, None)


class SimpleDetectionVisualizer(DetectionVisualizer):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if metric is not None and metric > self.threshold:
            draw_bbox(detection_pred, image, (0, 255, 0), 2, None, None, None, None)
        elif detection_pred is not None:
            draw_bbox(detection_pred, image, (255, 0, 0), 1, None, None, None, None)
            if detection_gt is not None:
                draw_bbox(detection_gt, image, (255, 0, 0), 1, None, None, None, None)
        elif detection_gt is not None:
            draw_bbox(detection_gt, image, (255, 0, 0), 1, None, None, None, None)


class EllipseDetectionVisualizer(DetectionVisualizer):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_gt is not None:
            color = (0,255,0)
            if (metric is not None and metric < self.threshold) or detection_pred is None:
                color = (255,0,0)
            x1, y1, x2, y2 = detection_gt.bbox.ltrb()
            center = (int((x1 + x2) / 2), int(y2))
            width = x2 - x1
            cv2.ellipse(
                image,
                center=center,
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=-45.0,
                endAngle=235.0,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4,
            )