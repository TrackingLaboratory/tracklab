import torch
import numpy as np
import pandas as pd

from pbtrack.core.tracker import OnlineTracker
from pbtrack.datastruct import Detections, ImageMetadatas, Detection, ImageMetadata
from pbtrack.utils.images import cv2_load_image
from plugins.track.strong_sort import StrongSORT


@torch.no_grad()
class StrongSORTTracker(OnlineTracker):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.reset()

    def reset(self):
        """ Reset the tracker state to start tracking in a new video."""
        self.model = StrongSORT(
            max_dist=self.cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=self.cfg.STRONGSORT.MAX_AGE,
            n_init=self.cfg.STRONGSORT.N_INIT,
            nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
        )
        # For camera compensation
        self.prev_frame = None

    def _camera_compensation(self, curr_frame):
        if self.cfg.STRONGSORT.ECC:  # camera motion compensation
            self.model.tracker.camera_update(self.prev_frame, curr_frame)
            self.prev_frame = curr_frame

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
                },
                index=results[:, -1].astype(int),
            )

            detections = detections.merge(
                track_df, left_index=True, right_index=True, validate="one_to_one"
            )
        else:  # FIXME
            detections["track_bbox_tlwh"] = np.nan
            detections["track_bbox_conf"] = np.nan
            detections["track_id"] = -1
        return detections
