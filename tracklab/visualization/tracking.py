import cv2
import numpy as np
import logging

from tracklab.visualization import DetectionVisualizer

log = logging.getLogger(__name__)

class TrackingLine(DetectionVisualizer):
    def __init__(self, max_length: int = 60, vertical_pos: float = 0.0):
        super().__init__()
        self.max_length = max_length
        self.vertical_pos = vertical_pos
        self.frame_id = []
        self.xy = []
        self.track_id = []

    def preproces(self, video_detections_pred, video_detections_gt, video_image_pred, video_image_gt):
        for (track_id, image_id) , detection in video_detections_pred.groupby(["track_id", "image_id"]):
            assert len(detection) <= 1, "frame with duplicate track_ids"
            frame_id = video_image_gt.loc[image_id].name
            self.frame_id.append(frame_id)
            xy = detection.iloc[0].bbox.xywh()[:2]
            xy[1] += self.vertical_pos * (detection.iloc[0].bbox.xywh()[3] / 2)
            self.xy.append(xy)
            self.track_id.append(track_id)

    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        self.current_frame_id = image_gt.id
        super().draw_frame(image, detections_pred, detections_gt, image_pred, image_gt)

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        if detection_pred is None:
            return

        frame_ids = np.array(self.frame_id)
        track_ids = np.array(self.track_id)
        xy = np.array(self.xy)
        filtering_mask = frame_ids >= self.current_frame_id - self.max_length
        filtering_mask &= frame_ids <= self.current_frame_id
        filtering_mask &= track_ids == detection_pred.track_id
        current_xy = xy[(frame_ids == self.current_frame_id)&(track_ids == detection_pred.track_id)]
        current_xys = xy[filtering_mask]
        color = self.color(detection_pred, is_prediction=True, color_type="bbox")
        if color:
            cv2.polylines(
                image,
                [current_xys.astype(np.int32)],
                False,
                color=color,
                thickness=2,
            )
            if len(current_xy) == 1:
                cv2.circle(
                    image,
                    current_xy[0].astype(np.int32),
                    radius=5,
                    color=color,
                    thickness=-1,
                )
