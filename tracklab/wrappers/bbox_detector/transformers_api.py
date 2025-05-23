from typing import Any

import torch
import pandas as pd
from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh


class RTDetr(ImageLevelModule):
    input_columns = []
    output_columns = ["image_id", "video_id", "category_id", "bbox_ltwh", "bbox_conf"]

    def __init__(self, device, batch_size, model_name, min_confidence, **kwargs):
        super().__init__(batch_size)
        self.device = device
        self.image_processor = RTDetrImageProcessor.from_pretrained(f"PekingU/{model_name}")
        if "v2" in model_name:
            self.model = RTDetrV2ForObjectDetection.from_pretrained(f"PekingU/{model_name}", device_map=device)
        else:
            self.model = RTDetrForObjectDetection.from_pretrained(f"PekingU/{model_name}", device_map=device)
        self.min_confidence = min_confidence
        self.id = 0

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images = self.image_processor(batch, return_tensors="pt")
        outputs = self.model(**images)
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=[batch.shape[1:3]]*batch.shape[0], threshold=self.min_confidence
        )
        detections = []
        for i, result in enumerate(results):
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                if label == 0:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadatas["id"].values[i],
                                bbox_ltwh=ltrb_to_ltwh(box.numpy(), (batch.shape[2], batch.shape[1])),
                                bbox_conf=score.item(),
                                video_id=metadatas["video_id"].values[i],
                                category_id=1,
                            ),
                            name=self.id
                        )
                    )
                    self.id += 1
        return detections
