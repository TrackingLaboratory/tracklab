from collections import deque

import cv2
import numpy as np

from tracklab.core.visualizer import DetectionVisualizer


class TrackingLineVisualizer(DetectionVisualizer):
    def __init__(self, max_length: int = 60):
        super().__init__()
        self.current_frame_id = 0
        self.max_length = max_length
        self.frame_id = []  # deque(maxlen=max_length) # np.array([], dtype=int)
        self.xy = []  # deque(maxlen=max_length) # np.empty((0, 2), dtype=np.float32)
        self.track_id = []  # deque(maxlen=max_length) # np.array([], dtype=int)

    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        super().draw_frame(image, detections_pred, detections_gt, image_pred, image_gt)
        self.current_frame_id += 1

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_pred is None:
            return
        self.frame_id.append(self.current_frame_id)
        self.track_id.append(detection_pred.track_id)
        xy = detection_pred.bbox.xywh()[:2]
        self.xy.append(xy)
        frame_ids = np.array(self.frame_id)
        track_ids = np.array(self.track_id)
        xy = np.array(self.xy)
        filtering_mask = frame_ids >= self.current_frame_id - self.max_length
        self.frame_id = list(frame_ids[filtering_mask])
        self.track_id = list(track_ids[filtering_mask])
        self.xy = list(xy[filtering_mask])
        current_xy = xy[track_ids == detection_pred.track_id]
        cv2.polylines(
            image,
            [current_xy.astype(np.int32)],
            False,
            color=(247, 207, 37),
            thickness=2,
        )
