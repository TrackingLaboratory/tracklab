import torch
import numpy as np
import pandas as pd
from pathlib import Path

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
from deep_oc_sort import ocsort

import logging

log = logging.getLogger(__name__)


class DeepOCSORT(ImageLevelModule):
    input_columns = [
        "bbox_ltwh",
        "bbox_conf",
        "category_id",
    ]
    output_columns = ["track_id", "track_bbox_ltwh", "track_bbox_conf"]

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        self.cfg = cfg
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = ocsort.OCSort(
            Path(self.cfg.model_weights),
            self.device,
            self.cfg.fp16,
            **self.cfg.hyperparams
        )

    @torch.no_grad()
    def preprocess(self, detection: pd.Series, metadata: pd.Series):
        ltrb = detection.bbox.ltrb()
        conf = detection.bbox.conf()
        cls = detection.category_id
        tracklab_id = detection.name
        return {
            "input": np.array(
                [ltrb[0], ltrb[1], ltrb[2], ltrb[3], conf, cls, tracklab_id]
            ),
        }

    @torch.no_grad()
    def process(self, batch, image, detections: pd.DataFrame):
        inputs = batch["input"]  # Nx7 [l,t,r,b,conf,class,tracklab_id]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        results = self.model.update(inputs, image)
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
