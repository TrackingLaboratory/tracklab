import numpy as np
import torch

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ["StrongSORT"]


class StrongSORT(object):
    def __init__(
        self, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100,
    ):

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("part_based", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init
        )

    def update(
        self,
        ids,
        bbox_xywh,
        reid_features,
        visibility_scores,
        confidences,
        classes,
        ori_img,
    ):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(
                ids[i],
                bbox_tlwh[i],
                conf,
                {
                    "reid_features": np.asarray(
                        reid_features[i].cpu(), dtype=np.float32
                    ),
                    "visibility_scores": np.asarray(
                        visibility_scores[i].cpu(), dtype=np.float32
                    ),
                },
            )
            for i, conf in enumerate(confidences)
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            det = track.last_detection_to_tlwh()

            t, l, w, h = det.tlwh
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([t, l, w, h, track_id, class_id, conf, det.id]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        else:
            outputs = np.empty((0, 8))
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h
