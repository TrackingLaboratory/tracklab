import torch
import numpy as np
import plugins.track.strong_sort as strong_sort

from pbtrack import OnlineTracker, Detections, ImageMetadatas, Detection, ImageMetadata

import logging

log = logging.getLogger(__name__)


class StrongSORT(OnlineTracker):
    def __init__(self, cfg, device):
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

    @torch.no_grad()
    def preprocess(self, detection: Detection, metadata: ImageMetadata):
        bbox_ltwh = detection.bbox_ltwh
        score = np.mean(detection.keypoints_xyc[:, 2])  # TODO put as Detection property
        reid_features = detection.embeddings  # .flatten()
        visibility_score = detection.visibility_scores
        id = detection.id
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

    @torch.no_grad()
    def process(self, batch, image, detections: Detections, metadatas: ImageMetadatas):
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
        detections = self._update_detections(results, detections)
        return detections

    def _update_detections(self, results, detections):
        assert results.index.isin(
            detections.index
        ).all(), "StrongSORT returned detections with unknown indices"
        merged_detections = detections.join(results, how="left")
        assert merged_detections.index.equals(detections.index), (
            "Merge with StrongSORT results failed, some "
            "detections were lost or added"
        )
        detections = merged_detections
        return detections
