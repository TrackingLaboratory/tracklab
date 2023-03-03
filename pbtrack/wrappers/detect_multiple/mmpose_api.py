import os
import torch
import requests
import numpy as np
from tqdm import tqdm

import mmcv
from mmpose.apis import init_pose_model, inference_bottom_up_pose_model

from pbtrack import ImageMetadata, ImageMetadatas, Detector, Detection
from pbtrack.utils.images import cv2_load_image

import logging

log = logging.getLogger(__name__)
mmcv.collect_env()


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class MMPose(Detector):
    collate_fn = collate_fn

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.check_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.model = init_pose_model(cfg.path_to_config, cfg.path_to_checkpoint, device)
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
        detections = []
        for image, shape, (_, metadata) in zip(images, shapes, metadatas.iterrows()):
            pose_results, _ = inference_bottom_up_pose_model(self.model, image)
            for pose in pose_results:
                if pose["score"] >= self.cfg.instance_min_confidence:
                    keypoints = self.sanitize_keypoints(pose["keypoints"], shape)
                    bbox = self.generate_bbox(keypoints, shape)
                    detections.append(
                        Detection.create(
                            image_id=metadata.id,
                            id=self.id,
                            bbox_ltwh=bbox,
                            bbox_c=pose["score"],
                            keypoints_xyc=keypoints,
                            video_id=metadata.video_id,
                            category_id=1,  # `person` class in posetrack
                        )
                    )
                    self.id += 1
        return detections

    def generate_bbox(self, keypoints, image_shape):
        # generate a bounding box ltwh from the keypoints
        # the bbox is then sanitized by cropping it to the image shape
        keypoints = keypoints[keypoints[:, 2] > 0]
        lt = np.amin(keypoints[:, :2], axis=0)
        rb = np.amax(keypoints[:, :2], axis=0)
        bbox_w = rb[0] - lt[0]
        bbox_h = rb[1] - lt[1]
        lt[0] -= bbox_w * self.cfg.bbox.left_right_extend_factor
        rb[0] += bbox_w * self.cfg.bbox.left_right_extend_factor
        lt[1] -= bbox_h * self.cfg.bbox.top_extend_factor
        rb[1] += bbox_h * self.cfg.bbox.bottom_extend_factor
        new_l = np.clip(np.round(lt[0]), 0, image_shape[0])
        new_t = np.clip(np.round(lt[1]), 0, image_shape[1])
        new_w = np.clip(np.round(rb[0] - lt[0]), 0, image_shape[0] - new_l)
        new_h = np.clip(np.round(rb[1] - lt[1]), 0, image_shape[1] - new_t)
        return np.array([new_l, new_t, new_w, new_h], dtype=int)

    def sanitize_keypoints(self, keypoints, shape):
        # Clip the keypoints to the image dimensions
        # and set the confidence to 0 if thresholds are not met
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, shape[0])
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, shape[1])
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
