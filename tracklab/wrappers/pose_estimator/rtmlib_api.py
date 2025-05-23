from typing import Any

import cv2
import pandas as pd
import numpy as np
import accelerate
import rtmlib

from hydra.utils import instantiate

from tracklab.utils.coordinates import generate_bbox_from_keypoints
from tracklab.pipeline import ImageLevelModule


class RTMPose(ImageLevelModule):
    input_columns = []
    output_columns = ["keypoints_xyc", "keypoints_conf"]

    def __init__(self, device, model, **kwargs):
        super().__init__(batch_size=1)
        self.device = device
        self.model = instantiate(model, device=self.device, backend='onnxruntime')

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return {}

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        image = cv2.imread(metadatas['file_path'].values[0])  # BGR not RGB !
        bboxes = detections.bbox.ltrb().values
        keypoints, scores = self.model(image, bboxes)
        detections["keypoints_xyc"] = list(np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1))
        detections["keypoints_conf"] = list(np.mean(scores, axis=1))
        return detections


class RTMO(ImageLevelModule):
    input_columns = []
    output_columns = ["image_id", "video_id", "category_id", "bbox_ltwh", "bbox_conf", "keypoints_xyc", "keypoints_conf"]

    def __init__(self, device, model, min_confidence, **kwargs):
        super().__init__(batch_size=1)
        self.device = device
        self.model = instantiate(model, device=self.device, backend='onnxruntime')
        self.min_confidence = min_confidence
        self.id = 0

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return {}

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        image = cv2.imread(metadatas['file_path'].values[0])  # BGR not RGB !
        shape = (image.shape[1], image.shape[0])
        keypoints, scores = self.model(image)
        detections = []
        for kps, score in zip(keypoints, scores):
            conf = np.mean(score)
            kps = np.concatenate([kps, score[..., np.newaxis]], axis=-1)
            if conf >= self.min_confidence:
                detections.append(
                    pd.Series(
                        dict(
                            image_id=metadatas["id"].values[0],
                            bbox_ltwh=generate_bbox_from_keypoints(kps, [ 0.1, 0.03, 0.1 ], shape),
                            bbox_conf=conf,
                            keypoints_xyc=kps,
                            keypoints_conf=conf,
                            video_id=metadatas["video_id"].values[0],
                            category_id=1,
                        ),
                        name=self.id
                    )
                )
                self.id += 1
        return detections
