import cv2
import numpy as np

from tracklab.visualization import DetectionVisualizer
from tracklab.utils.cv2 import draw_bbox, draw_bbox_stats, draw_text


class DefaultDetection(DetectionVisualizer):
    def __init__(self, print_id=True, print_confidence=False):
        super().__init__()
        self.print_id = print_id
        self.print_confidence = print_confidence

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        for detection, is_pred in zip([detection_pred, detection_gt], [True, False]):
            if detection is not None:
                color_bbox = self.color(detection, is_prediction=is_pred)
                if color_bbox:
                    draw_bbox(
                        detection,
                        image,
                        color_bbox,
                        print_id=self.print_id,
                        print_confidence=self.print_confidence,
                    )

class FullDetection(DefaultDetection):
    def __init__(self):
        super().__init__(print_id=True, print_confidence=True)

class DebugDetection(DetectionVisualizer):
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

class DetectionStats(DetectionVisualizer):
    def __init__(self,
            print_stats=["state", "hits", "age", "time_since_update", "matched_with"],  # FIXME "costs" is too long for display
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
                    self.print_stats,
                    bbox_color=color_bbox,
                )

class SimpleDetectionStats(DetectionStats):
    def __init__(self):
        super().__init__(print_stats=["state", "hits", "age", "time_since_update"])

class EllipseDetection(DetectionVisualizer):
    def __init__(self, print_id=True):
        self.print_id = print_id
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        for detection, is_pred in zip([detection_pred, detection_gt], [True, False]):
            if detection is not None:
                color = self.color(detection, is_prediction=is_pred)
                if color:
                    x1, y1, x2, y2 = detection.bbox.ltrb()
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
                        lineType=cv2.LINE_AA,
                    )
                    if self.print_id and hasattr(detection, "track_id"):
                        draw_text(
                            image,
                            f"ID: {int(detection.track_id)}",
                            center,
                            fontFace=1,
                            fontScale=1,
                            thickness=1,
                            alignH="c",
                            alignV="c",
                            color_bg=color,
                            color_txt=None,
                            alpha_bg=1,
                        )
