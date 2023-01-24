import numpy as np
import torch

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ["StrongSORT"]


class StrongSORT(object):
    def __init__(
        self,
        ema_alpha=0.9,
        mc_lambda=0.995,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100,
        min_bbox_confidence=0.2,
    ):

        self.max_dist = max_dist
        self.min_bbox_confidence = min_bbox_confidence
        metric = NearestNeighborDistanceMetric("part_based", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            ema_alpha=ema_alpha,
            mc_lambda=mc_lambda,
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
        frame
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
                        visibility_scores[i].cpu()
                    ),
                },
            )
            for i, conf in enumerate(confidences)
        ]

        detections = self.filter_detections(detections)

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                # Vlad: Before 'track.time_since_update > 0', it was 'track.time_since_update > 1', which means that a
                # track that was not updated at the current frame but was updated at the previous frame would still be
                # returned. Because of this, the last detection in that track would be outputted twice in a row,
                # which is not what we want. The original StrongSORT implementation was not returning the last detection
                # in the tracklet, but rather the predicted bbox in the current frame (computed using the Kalman filter
                # and the tracklet history). Consequently, it could have made sense to output the predicted bbox here
                # even if the tracklet was not updated, but why doing it onlu for tracklets with
                # track.time_since_update = 1? Why not put that value "1" as a parameter of the tracker?
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

    def filter_detections(self, detections):
        detections = [det for det in detections if det.confidence > self.min_bbox_confidence]
        return detections
