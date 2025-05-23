from typing import Any

import torch
import pandas as pd
import numpy as np
from transformers import AutoProcessor, VitPoseForPoseEstimation

from tracklab.pipeline import DetectionLevelModule


class VITPose(DetectionLevelModule):
    input_columns = []
    output_columns = ["keypoints_xyc", "keypoints_conf"]

    def __init__(self, device, batch_size, model_name, **kwargs):
        super().__init__(batch_size)
        self.device = device
        self.image_processor = AutoProcessor.from_pretrained(f"usyd-community/{model_name}")
        self.model = VitPoseForPoseEstimation.from_pretrained(f"usyd-community/{model_name}", device_map=device)

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series) -> Any:
        return {"image": image, "bbox": detection["bbox_ltwh"]}

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        boxes = batch["bbox"].unsqueeze(1)
        inputs = self.image_processor(batch["image"], boxes=boxes, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, dataset_index=torch.tensor(boxes.shape[0]*[0], device=self.device))
        pose_results = self.image_processor.post_process_pose_estimation(outputs, boxes=boxes)
        keypoints_xy = np.array([res[0]["keypoints"].cpu().numpy() for res in pose_results])
        keypoints_c = np.array([res[0]["scores"].cpu().numpy() for res in pose_results])
        detections["keypoints_xyc"] = list(np.concatenate([keypoints_xy, keypoints_c[..., np.newaxis]], axis=-1))
        detections["keypoints_conf"] = list(keypoints_c.mean(axis=1))
        return detections
