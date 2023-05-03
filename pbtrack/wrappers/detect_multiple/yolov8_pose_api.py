import os
import torch
import numpy as np
import pandas as pd

os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

from pbtrack.pipeline import MultiDetector
from pbtrack.utils.cv2 import cv2_load_image
from pbtrack.utils.coordinates import ltrb_to_ltwh

import logging

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOv8Pose(MultiDetector):
    collate_fn = collate_fn

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        self.model = YOLO(cfg.path_to_checkpoint)
        self.model.to(device)
        self.id = 0

    @torch.no_grad()
    def preprocess(self, metadata: pd.Series):
        image = cv2_load_image(metadata.file_path)
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch, metadatas: pd.DataFrame):
        images, shapes = batch
        results_by_image = self.model(images)
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
                                keypoints_xyc=keypoints,
                                keypoints_conf=np.mean(keypoints[:, 2], axis=0),
                                track_id=np.nan,
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections
