import copy
from functools import partial
import os
import cv2
import torch
import requests
import numpy as np
from tqdm import tqdm

import mmcv
from mmpose.apis import init_pose_model
from mmcv.parallel import collate, scatter
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose, ToTensor


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


def mmpose_collate(batch):
    return collate(batch, len(batch))

class MMPose(Detector):
    collate_fn = mmpose_collate

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.check_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.model = init_pose_model(cfg.path_to_config, cfg.path_to_checkpoint, device)
        self.device = device
        
        cfg = self.model.cfg
        self.dataset_info = DatasetInfo(cfg.dataset_info)
        self.test_pipeline = Compose(cfg.test_pipeline)
        # _pipeline_gpu_speedup(self.test_pipeline, device)

        self.flip_pairs = self.dataset_info.flip_pairs
        self.collate_fn = partial(self.collate_fn, samples_per_gpu=32)


    @torch.no_grad()
    def preprocess(self, detection: Detection, metadata: ImageMetadata):
        cfg = self.model.cfg
        image = cv2_load_image(metadata.file_path)
        ltwh = detection.bbox_ltwh
        ltrb = self.sanitize_bbox(ltwh, (image.shape[1], image.shape[0]))
        crop = image[ltrb[1] : ltrb[3], ltrb[0] : ltrb[2]]
        bbox = np.array([0, 0, crop.shape[1], crop.shape[0]])
        data = {
            "bbox": bbox,
            "img": crop,
            "bbox_score": 1.0,
            "bbox_id": 0,
            "dataset": self.dataset_info.dataset_name,
            "joints_3d": np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            "joints_3d_visible": np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            "rotation": 0,
            "ann_info": {
                "image_size": np.array(cfg.data_cfg['image_size']),
                "num_joints": cfg.data_cfg.num_joints,
                "flip_pairs": self.flip_pairs,             
            },
        }
        data = self.test_pipeline(data)
        data["img_metas"]._data["bbox_offset"] = ltrb[:2]
        data["img_metas"]._data["shape"] = (image.shape[1], image.shape[0])
        return data

    @torch.no_grad()
    def process(self, batch, detections: Detections, metadatas: ImageMetadatas):
        batch = scatter(batch, [self.device])[0]
        with torch.no_grad():
            kp_results = self.model(
                img=batch["img"],
                img_metas=batch["img_metas"],
                return_loss=False,
                return_heatmap=False,
            )["preds"]
        results = []
        for result, img_metas in zip(kp_results, batch["img_metas"]):
            bbox_offset = img_metas["bbox_offset"]
            shape = img_metas["shape"]
            results.append(
                self.sanitize_keypoints(result, bbox_offset, shape)
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
            log.info("Checkpoint not found at {}".format(path_to_checkpoint))
            log.info("Downloading checkpoint from {}".format(download_url))
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
                log.warning(
                    f"Something went wrong while downloading or writing {download_url} to {path_to_checkpoint}"
                )
            else:
                log.info("Checkpoint downloaded successfully")



def _pipeline_gpu_speedup(pipeline, device):
    for t in pipeline.transforms:
        if isinstance(t, ToTensor):
            t.device = device
