import os
import torch
import requests
import numpy as np
from tqdm import tqdm

import mmcv
from mmpose.apis import init_pose_model, inference_top_down_pose_model

from pbtrack import ImageMetadata, Detections, ImageMetadatas, Detector, Detection
from pbtrack.utils.images import cv2_load_image

import logging

log = logging.getLogger(__name__)
mmcv.collect_env()


def collate_fn(batch):
    idxs = [b for b, _ in batch]
    crops = [b["crop"] for _, b in batch]
    bboxes = [b["bbox_offset"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (crops, bboxes, shapes)


class MMPose(Detector):
    collate_fn = collate_fn

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.check_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.model = init_pose_model(cfg.path_to_config, cfg.path_to_checkpoint, device)
        self.device = device

    @torch.no_grad()
    def preprocess(self, detection: Detection, metadata: ImageMetadata):
        image = cv2_load_image(metadata.file_path)
        ltwh = detection.bbox_ltwh
        ltrb = self.sanitize_bbox(ltwh, (image.shape[1], image.shape[0]))
        crop = image[ltrb[1] : ltrb[3], ltrb[0] : ltrb[2]]
        return {
            "crop": crop,
            "bbox_offset": ltrb[:2],
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch, detections: Detections, metadatas: ImageMetadatas):
        crops, bbox_offsets, shapes = batch
        results = []
        for crop, bbox_offset, shape in zip(crops, bbox_offsets, shapes):
            result, _ = inference_top_down_pose_model(self.model, crop)
            assert len(result) == 1
            results.append(
                self.sanitize_keypoints(result[0]["keypoints"], bbox_offset, shape)
            )
        detections["keypoints_xyc"] = results
        return detections

    @staticmethod
    def sanitize_bbox(bbox, image_shape):
        # convert ltwh bbox to ltrb,check if bbox coordinates are within
        # image dimensions and round them to int
        new_l = np.clip(np.round(bbox[0]), 0, image_shape[0])
        new_t = np.clip(np.round(bbox[1]), 0, image_shape[1])
        new_r = np.clip(np.round(bbox[2] + new_l), 0, image_shape[0])
        new_b = np.clip(np.round(bbox[3] + new_t), 0, image_shape[1])
        return np.array([new_l, new_t, new_r, new_b], dtype=int)

    def sanitize_keypoints(self, keypoints, bbox_offset, shape):
        # apply the offset to the keypoints, clip them to the image dimensions
        # and set the confidence to 0 if thresholds are not met
        keypoints[:, :2] += bbox_offset
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, shape[0])
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, shape[1])
        if np.mean(keypoints[:, 2]) < self.cfg.instance_min_confidence:
            keypoints[:, 2] = 0.0
        keypoints[keypoints[:, 2] < self.cfg.keypoint_min_confidence, 2] = 0.0
        return keypoints

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
