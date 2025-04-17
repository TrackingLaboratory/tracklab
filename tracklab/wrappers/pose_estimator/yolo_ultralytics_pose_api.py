import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
from ultralytics import YOLO

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOUltralyticsPose(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
        "keypoints_xyc",
        "keypoints_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.path_to_checkpoint)
        self.model.to(device)
        self.id = 0

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self,  batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        results_by_image = self.model(images, verbose=False)
        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox, keypoints in zip(
                results.boxes.cpu().numpy(), results.keypoints.cpu().numpy()
            ):
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                                keypoints_xyc=keypoints.data[0],
                                keypoints_conf=np.mean(keypoints.data[0, :, 2]),
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections
