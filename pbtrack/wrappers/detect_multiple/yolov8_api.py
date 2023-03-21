import torch
import numpy as np

from ultralytics import YOLO

from pbtrack import ImageMetadata, ImageMetadatas, Detector, Detection
from pbtrack.utils.images import cv2_load_image

import logging

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOv8(Detector):
    collate_fn = collate_fn

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.model = YOLO(cfg.model_name)
        self.model.to(device)
        self.device = device
        self.id = 0

    @torch.no_grad()
    def preprocess(self, metadata: ImageMetadata):
        image = cv2_load_image(metadata.file_path)
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, preprocessed_batch: dict, metadatas: ImageMetadatas):
        images, shapes = preprocessed_batch
        results_by_image = self.model(images)
        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes:
                # check for `person` class
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_bbox_score:
                    detections.append(
                        Detection.create(
                            image_id=metadata.id,
                            id=self.id,
                            bbox_ltwh=self.sanitize_bbox(bbox.xyxy, shape),
                            bbox_score=bbox.conf.item(),
                            video_id=metadata.video_id,
                            category_id=1,  # `person` class in posetrack
                        )
                    )
                    self.id += 1
        return detections

    @staticmethod
    def sanitize_bbox(bbox, image_shape):
        # from ltrb to ltwh sanitized
        # bbox coordinates rounded as int and clipped to the image size
        bbox = bbox[0].cpu()
        new_l = np.clip(np.round(bbox[0]), 0, image_shape[0])
        new_t = np.clip(np.round(bbox[1]), 0, image_shape[1])
        new_w = np.clip(np.round(bbox[2] - bbox[0]), 0, image_shape[0] - new_l)
        new_h = np.clip(np.round(bbox[3] - bbox[1]), 0, image_shape[1] - new_t)
        return np.array([new_l, new_t, new_w, new_h], dtype=int)
