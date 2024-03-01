import torch
import numpy as np
import pandas as pd

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
import oc_sort.ocsort as ocsort

import logging

log = logging.getLogger(__name__)


class OCSORT(ImageLevelModule):
    input_columns = [
        "bbox_ltwh",
        "bbox_conf",
        "category_id",
    ]
    output_columns = ["track_id", "track_bbox_ltwh", "track_bbox_conf"]

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)  # Fixed batch size of 1 for trackers
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = ocsort.OCSort(**self.cfg.hyperparams)

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        processed_detections = []
        if len(detections) == 0:
            return {"input": []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)
            processed_detections.append(
                np.array([*ltrb, conf, cls, tracklab_id])
            )
        return {
            "input": np.stack(processed_detections),
        }

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return []
        inputs = batch["input"][0]  # Nx7 [l,t,r,b,conf,class,tracklab_id]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        results = self.model.update(inputs, None)
        results = np.asarray(results)  # N'x8 [l,t,r,b,track_id,class,conf,idx]
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 7].astype(int))
            assert set(idxs).issubset(
                detections.index
            ), "Mismatch of indexes during the tracking. The results should match the detections."
            results = pd.DataFrame(
                {
                    "track_bbox_ltwh": track_bbox_ltwh,
                    "track_bbox_conf": track_bbox_conf,
                    "track_id": track_ids,
                    "idxs": idxs,
                }
            )
            results.set_index("idxs", inplace=True, drop=True)
            return results
        else:
            return []
