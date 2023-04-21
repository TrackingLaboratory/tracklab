import pandas as pd
import torch
import numpy as np
import plugins.track.strong_sort as strong_sort

from pbtrack.pipeline import Tracker

import logging

log = logging.getLogger(__name__)

@torch.no_grad()
class StrongSORT(Tracker):
    input_columns = ["bbox_ltwh", "keypoints_xyc", "keypoints_score",
                     "embeddings", "visibility_scores"]
    output_columns = ["track_id",
                      "track_bbox_kf_ltwh", "track_bbox_pred_kf_ltwh",
                      "matched_with", "costs", "hits",
                      "age", "time_since_update", "state"]

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        self.cfg = cfg
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = strong_sort.StrongSORT(
            ema_alpha=self.cfg.ema_alpha,
            mc_lambda=self.cfg.mc_lambda,
            max_dist=self.cfg.max_dist,
            motion_criterium=self.cfg.motion_criterium,
            max_iou_distance=self.cfg.max_iou_distance,
            max_oks_distance=self.cfg.max_oks_distance,
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            nn_budget=self.cfg.nn_budget,
            min_bbox_confidence=self.cfg.min_bbox_confidence,
            only_position_for_kf_gating=self.cfg.only_position_for_kf_gating,
            max_kalman_prediction_without_update=self.cfg.max_kalman_prediction_without_update,
            matching_strategy=self.cfg.matching_strategy,
            gating_thres_factor=self.cfg.gating_thres_factor,
            w_kfgd=self.cfg.w_kfgd,
            w_reid=self.cfg.w_reid,
            w_st=self.cfg.w_st,
        )
        # For camera compensation
        self.prev_frame = None
        self.failed_ecc_counter = 0

    def prepare_next_frame(self, next_frame: np.ndarray):
        # Propagate the state distribution to the current time step using a Kalman filter prediction step.
        self.model.tracker.predict()

        # Camera motion compensation
        if self.cfg.ecc:
            if self.prev_frame is not None:
                matrix = self.model.tracker.camera_update(self.prev_frame, next_frame)
                if matrix is None:
                    self.failed_ecc_counter += 1
            self.prev_frame = next_frame

    def preprocess(self, detection: pd.Series, metadata: pd.Series):
        bbox_ltwh = detection.bbox_ltwh
        score = detection.keypoints_conf
        reid_features = detection.embeddings  # .flatten()
        visibility_score = detection.visibility_scores
        id = detection.name
        classes = np.array(0)
        keypoints = detection.keypoints_xyc
        return (
            id,
            bbox_ltwh,
            reid_features,
            visibility_score,
            score,
            classes,
            metadata.frame,
            keypoints,
        )

    def process(self, batch, image, detections: pd.DataFrame):
        (
            id,
            bbox_ltwh,
            reid_features,
            visibility_scores,
            scores,
            classes,
            frame,
            keypoints,
        ) = batch
        results = self.model.update(
            id,
            bbox_ltwh,
            reid_features,
            visibility_scores,
            scores,
            classes,
            image,
            frame,
            keypoints,
        )
        return results
