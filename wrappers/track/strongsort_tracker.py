import torch
import numpy as np
import pandas as pd

from pathlib import Path
from pbtrack.core.tracker import OnlineTracker
from pbtrack.datastruct import Detections, ImageMetadatas, Detection, ImageMetadata
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
        image = metadata.image
        self._camera_compensation(image)
        bbox = detection.bbox_cmwh
        score = np.mean(detection.keypoints_xyc[:, 2])
        reid_features = detection.embeddings  # .flatten()
        visibility_score = detection.visibility_scores
        classes = np.array(0)
        return bbox, reid_features, visibility_score, score, classes, image

    def process(self, batch, detections: Detections, metadatas: ImageMetadatas):
        cmwhs, reid_features, visibility_scores, scores, classes, image = batch
        if cmwhs.numel() != 0:
            results = self.model.update(
                cmwhs, reid_features, visibility_scores, scores, classes, image
            )
            print(results)
            detections = self._update_detections(results, detections)
        return detections

    def _update_detections(self, results, detections):
        track_bboxes = []
        track_bbox_confs = []
        person_id = []
        for result in results:
            detection = detections.iloc[int(result[-1])]  # result[:4], detections)
            w = result[2] - result[0]
            h = result[3] - result[1]
            detection.track_bbox = [result[0] + w / 2, result[1] + h / 2, w, h]
            detection.track_bbox_conf = result[6]
            detection.person_id = int(result[4])
            track_df = pd.DataFrame(
                dict(
                    track_bbox=[result[0] + w / 2, result[1] + h / 2, w, h],
                    track_bbox_conf=result[6],
                    person_id=int(result[4]),
                )
            )
            detections = detections.merge(
                track_df, left_index=True, right_index=True, validate="one_to_one"
            )
        return detections
