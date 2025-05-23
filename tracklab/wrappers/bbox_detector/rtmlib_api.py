from typing import Any

import accelerate
import rtmlib
import cv2
import pandas as pd

from hydra.utils import instantiate

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh


class RTMLibDetector(ImageLevelModule):
    input_columns = []
    output_columns = ["image_id", "video_id", "category_id", "bbox_ltwh", "bbox_conf"]

    def __init__(self, device, model, **kwargs):
        super().__init__(batch_size=1)
        self.device = device
        self.model = instantiate(model, device=self.device, backend='onnxruntime')
        self.id = 0

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return {}

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        image = cv2.imread(metadatas['file_path'].values[0])  # BGR not RGB !
        shape = (image.shape[1], image.shape[0])
        bboxes = self.model(image)
        detections = []
        for bbox in bboxes:
            detections.append(
                pd.Series(
                    dict(
                        image_id=metadatas["id"].values[0],
                        bbox_ltwh=ltrb_to_ltwh(bbox, shape),
                        bbox_conf=1.0,
                        video_id=metadatas["video_id"].values[0],
                        category_id=1,
                    ),
                    name=self.id
                )
            )
            self.id += 1
        return detections
