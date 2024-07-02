from typing import Any

import pandas as pd

from tracklab.pipeline import ImageLevelModule
from tracklab.utils import add_negative_samples


class NegativeKeypoints(ImageLevelModule):
    input_columns = ["keypoints_xyc"]
    output_columns = ["negative_kps"]
    def __init__(self, **kwargs):
        super().__init__(batch_size=1)

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return []

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        detections["negative_kps"] = \
        detections.groupby("image_id").apply(add_negative_samples).reset_index(level=0,
                                                                               drop=True)[
            "negative_kps"]
        return detections
