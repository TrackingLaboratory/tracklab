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
        only_position_for_kf_gating=False,
        max_kalman_prediction_without_update=7,
        matching_strategy="strong_sort_matching",
        gating_thres_factor=1.5,
        w_kfgd=1,
        w_reid=1,
        w_st=1,
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
            only_position_for_kf_gating=only_position_for_kf_gating,
            max_kalman_prediction_without_update=max_kalman_prediction_without_update,
            matching_strategy=matching_strategy,
            gating_thres_factor=gating_thres_factor,
            w_kfgd=w_kfgd,
            w_reid=w_reid,
            w_st=w_st,
        )

    def update(
        self,
        ids,
        bbox_ltwh,
        reid_features,
        visibility_scores,
        confidences,
        classes,
        frame,
        keypoints=None,
    ):
        # generate detections
        detections = [
            Detection(
                ids[i].cpu().detach().numpy(),
                np.asarray(bbox_ltwh[i].cpu().detach().numpy(), dtype=float),
                conf.cpu().detach().numpy(),
                {
                    "reid_features": np.asarray(
                        reid_features[i].cpu().detach().numpy(), dtype=np.float32
                    ),
                    "visibility_scores": np.asarray(
                        visibility_scores[i].cpu().detach().numpy()
                    ),
                },
                keypoints=keypoints[i].cpu().detach().numpy() if keypoints is not None else None,
            )
            for i, conf in enumerate(confidences)
        ]

        detections = self.filter_detections(detections)

        # update tracker
        self.tracker.predict()
        assert self.tracker.predict_done, "predict() must be called before update()"
        if len(detections) > 0:
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

            # TODO should update all detections, and set default values for non match (e.g. -1)
            det = track.last_detection
            # KF predicted bbox to be stored next to actual bbox :
            result_det = {  # if keys are added/updated here, don't forget to update the columns in the pd.DataFrame below
                "track_id": track.track_id,
                "track_bbox_kf_ltwh": track.to_ltwh(),
                "track_bbox_pred_kf_ltwh": track.last_kf_pred_ltwh,
                "matched_with": det.matched_with,
                "costs": det.costs,
                "hits": track.hits,
                "age": track.age,
                "time_since_update": track.time_since_update,
                "state": track.state,
            }
            ids.append(det.id)
            outputs.append(result_det)
        # FIXME I do not like to use a pandas dataframe here since it brings some hacky code to handle
        # it in the API. I would rather use a list of dicts ? For me it should be general here and then we do the
        # plumbery in the API
        outputs = pd.DataFrame(
            outputs,
            index=np.array(ids),
            columns=[
                "track_id",
                "track_bbox_kf_ltwh",
                "track_bbox_pred_kf_ltwh",
                "matched_with",
                "costs",
                "hits",
                "age",
                "time_since_update",
                "state",
            ],
        )
        return outputs

    def filter_detections(self, detections):
        detections = [
            det for det in detections if det.confidence > self.min_bbox_confidence
        ]
        return detections
