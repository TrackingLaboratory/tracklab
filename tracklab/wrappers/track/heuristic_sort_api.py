import torch
import pandas as pd

from tracklab.pipeline import ImageLevelModule
from heuristic_sort.heuristic_sort import HeuristicSORT

import logging

log = logging.getLogger(__name__)


class HeuristicSORTAPI(ImageLevelModule):
    input_columns = ["keypoints_xyc", "embeddings", "visibility_scores"]
    output_columns = [
        "track_id",
        "last_keypoints_xyc",
        "last_center",
        "pred_center",
        "update_center",
    ]

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = HeuristicSORT(**self.cfg)

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        return image


    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return pd.DataFrame(columns=["track_id"])
        image = batch[0].cpu().numpy()
        keypoints = detections["keypoints_xyc"].to_numpy() # batch["keypoints"].cpu().numpy()
        reid_features = detections["embeddings"].to_numpy() #  batch["reid_features"].cpu().numpy()
        visibility_scores = detections["visibility_scores"].to_numpy()
        pbtrack_ids = detections.index.to_numpy()
        results = self.model.update(
            keypoints, reid_features, visibility_scores, pbtrack_ids, image
        )
        if results:
            results = pd.DataFrame(results)
            results.set_index("pbtrack_id", inplace=True, drop=True)
            assert set(results.index).issubset(
                detections.index
            ), "Mismatch of indexes during the tracking. The results should match the detections."
            return results
        else:
            return pd.DataFrame(columns=["track_id"])
