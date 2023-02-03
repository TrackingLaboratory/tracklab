import numpy as np
import pandas as pd

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
        motion_criterium="iou",
        max_iou_distance=0.7,
        max_oks_distance=0.7,
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
            motion_criterium=motion_criterium,
            max_iou_distance=max_iou_distance,
            max_oks_distance=max_oks_distance,
            max_age=max_age,
            n_init=n_init,
            ema_alpha=ema_alpha,
            mc_lambda=mc_lambda,
        )

    def update(
        self,
        ids,
        bbox_ltwh,
        reid_features,
        visibility_scores,
        confidences,
        classes,
        ori_img,
        frame,
        keypoints
    ):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        detections = [
            Detection(
                ids[i].numpy(),
                np.asarray(bbox_ltwh[i], dtype=np.float),
                conf,
                {
                    "reid_features": np.asarray(
                        reid_features[i].cpu(), dtype=np.float32
                    ),
                    "visibility_scores": np.asarray(
                        visibility_scores[i].cpu()
                    ),
                },
                keypoints=keypoints[i].cpu().numpy()
            )
            for i, conf in enumerate(confidences)
        ]

        detections = self.filter_detections(detections)

        # update tracker
        assert self.tracker.predict_done, "predict() must be called before update()"
        self.tracker.update(detections, classes, confidences)
        self.tracker.predict_done = False

        # output bbox identities
        outputs = []
        ids = []
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

            det = track.last_detection
            # l, t, w, h = det.tlwh
            # KF predicted bbox to be stored next to actual bbox :
            l, t, w, h = track.to_tlwh()  # FIXME called 'tlwh' in StrongSORT, but contains actually 'ltwh'
            result_det = {
                "track_id": track.track_id,
                "track_bbox_ltwh": np.array([l, t, w, h]),
                "track_bbox_conf": track.conf,
                "matched_with": det.matched_with,
                "costs": det.costs,
            }
            ids.append(det.id)
            outputs.append(result_det)
        outputs = pd.DataFrame(outputs,
                               index=np.array(ids),
                               columns=['track_id', 'track_bbox_ltwh', 'track_bbox_conf', 'matched_with', 'costs'])
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    # @staticmethod
    # def _xywh_to_tlwh(bbox_xywh):  # BAD NAME, should be ltwh right?
    #     if isinstance(bbox_xywh, np.ndarray):
    #         bbox_tlwh = bbox_xywh.copy()
    #     elif isinstance(bbox_xywh, torch.Tensor):
    #         bbox_tlwh = bbox_xywh.clone()
    #     bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
    #     bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
    #     return bbox_tlwh

    def filter_detections(self, detections):
        detections = [det for det in detections if det.confidence > self.min_bbox_confidence]
        return detections
