import os
import torch
import requests
import numpy as np
from tqdm import tqdm

import mmcv
from mmdet.apis import init_detector, inference_detector

from pbtrack import ImageMetadatas, ImageMetadata, Detector, Detection
from pbtrack.utils.images import cv2_load_image

import logging

log = logging.getLogger(__name__)
mmcv.collect_env()


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class MMDetection(Detector):
    collate_fn = collate_fn

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.check_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.model = init_detector(cfg.path_to_config, cfg.path_to_checkpoint, device)
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
        results = inference_detector(self.model, images)

        detections = []
        for predictions, shape, (_, metadata) in zip(
            results, shapes, metadatas.iterrows()
        ):
            for prediction in predictions[0]:  # only check for 'person' class
                if prediction[4] >= self.cfg.bbox_min_confidence:
                    detections.append(
                        Detection.create(
                            image_id=metadata.id,
                            id=self.id,
                            bbox_ltwh=self.sanitize_bbox(prediction[:4], shape),
                            bbox_c=prediction[4],
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
        new_l = np.clip(np.round(bbox[0]), 0, image_shape[0])
        new_t = np.clip(np.round(bbox[1]), 0, image_shape[1])
        new_w = np.clip(np.round(bbox[2] - bbox[0]), 0, image_shape[0] - new_l)
        new_h = np.clip(np.round(bbox[3] - bbox[1]), 0, image_shape[1] - new_t)
        return np.array([new_l, new_t, new_w, new_h], dtype=int)

    @staticmethod
    def check_checkpoint(path_to_checkpoint, download_url):
        os.makedirs(os.path.dirname(path_to_checkpoint), exist_ok=True)
        if not os.path.exists(path_to_checkpoint):
            print("Checkpoint not found at {}".format(path_to_checkpoint))
            print("Downloading checkpoint from {}".format(download_url))
            response = requests.get(download_url, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            with open(path_to_checkpoint, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong while downloading or writing.")
            else:
                print("Checkpoint downloaded successfully")
