import torch
import numpy as np
import pandas as pd

from pbtrack.core.tracker import OnlineTracker
from pbtrack.core.datastruct import Detections, ImageMetadatas, Detection, ImageMetadata
from pbtrack.utils.images import cv2_load_image

import plugins.track.strong_sort as strong_sort


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
            max_iou_distance=self.cfg.max_iou_distance,
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            nn_budget=self.cfg.nn_budget,
        )
        # For camera compensation
        self.prev_frame = None

    def _camera_compensation(self, curr_frame):
        if self.cfg.ecc:  # camera motion compensation
            self.model.tracker.camera_update(self.prev_frame, curr_frame)
            self.prev_frame = curr_frame

    @torch.no_grad()
    def preprocess(self, detection: Detection, metadata: ImageMetadata):
        image = cv2_load_image(metadata.file_path)
        self._camera_compensation(image)
        bbox = detection.bbox_cmwh
        score = np.mean(detection.keypoints_xyc[:, 2])
        reid_features = detection.embeddings  # .flatten()
        visibility_score = detection.visibility_scores
        id = detection.id
        classes = np.array(0)
        return id, bbox, reid_features, visibility_score, score, classes, image

    @torch.no_grad()
    def process(self, batch, detections: Detections, metadatas: ImageMetadatas):
        id, cmwhs, reid_features, visibility_scores, scores, classes, image = batch
        if cmwhs.numel() != 0:
            results = self.model.update(
                id, cmwhs, reid_features, visibility_scores, scores, classes, image
            )
            detections = self._update_detections(results, detections)
        return detections

    def _update_detections(self, results, detections):
        if results.any():
            track_df = pd.DataFrame(
                {
                    "track_bbox_tlwh": list(results[:, 0:4]),
                    "track_bbox_conf": results[:, 6],
                    "track_id": results[:, 4].astype(int),
                    "person_id": results[:, 4].astype(int),
                },
                index=results[:, -1].astype(int),
            )

            detections = detections.merge(
                track_df, left_index=True, right_index=True, validate="one_to_one"
            )
        else:  # FIXME
            detections["track_bbox_tlwh"] = pd.NA
            detections["track_bbox_conf"] = pd.NA
            detections["track_id"] = pd.NA
        return detections
