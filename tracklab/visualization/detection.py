import cv2
import numpy as np

from tracklab.visualization import DetectionVisualizer
from tracklab.utils.cv2 import draw_bbox, draw_bbox_stats


class DefaultDetectionVisualizer(DetectionVisualizer):
    def __init__(self, print_id=True, print_confidence=False):
        super().__init__()
        self.print_id = print_id
        self.print_confidence = print_confidence

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_gt is not None:
            color_bbox = self.color(detection_gt, is_prediction=False)
            if color_bbox:
                draw_bbox(
                    detection_gt,
                    image,
                    color_bbox,
                )
        if detection_pred is not None:
            color_bbox = self.color(detection_pred, is_prediction=True)
            if color_bbox:
                draw_bbox(
                    detection_pred,
                    image,
                    color_bbox,
                    print_id=self.print_id,
                    print_confidence=self.print_confidence,
                )

class FullDetectionVisualizer(DefaultDetectionVisualizer):
    def __init__(self):
        super().__init__(print_id=True, print_confidence=True)

class DebugDetectionVisualizer(DetectionVisualizer):
    """
    Detections are classified by colors:
        - Green is True Positive
        - Yellow is False Positive
        - Red is False Negative
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_gt is not None:  # GT exists
            if detection_pred is None:  # pred is not detected
                draw_bbox(detection_gt, image, (255, 0, 0))  # FN
            elif metric and metric > self.threshold and not np.isnan(detection_pred.track_id):  # pred is correct
                draw_bbox(detection_pred, image, (0, 255, 0))  # TP
            else:  # pred is not correct
                draw_bbox(detection_gt, image, (255, 0, 0))  # FN
        elif detection_pred is not None and not np.isnan(detection_pred.track_id):  # no GT and pred is assigned
            draw_bbox(detection_pred, image, (255, 255, 0))  # FP

class DetectionStatsVisualizer(DetectionVisualizer):
    def __init__(self,
            print_stats=["state", "hits", "age", "time_since_update", "matched_with", "costs"],
     ):
        self.print_stats = print_stats
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_pred is not None:
            color_bbox = self.color(detection_pred, is_prediction=True)
            if color_bbox:
                draw_bbox_stats(
                    detection_pred,
                    image,
                    self.print_stats
                )

class SimpleDetectionStatsVisualizer(DetectionStatsVisualizer):
    def __init__(self):
        super().__init__(print_stats=["state", "hits", "age", "time_since_update"])

class EllipseDetectionVisualizer(DetectionVisualizer):
    def __init__(self):
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_gt is not None:
            color = self.color(detection_gt, is_prediction=False)
            if color:
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
        if detection_pred is not None:
            color = self.color(detection_pred, is_prediction=True)
            if color:
                x1, y1, x2, y2 = detection_pred.bbox.ltrb()
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
