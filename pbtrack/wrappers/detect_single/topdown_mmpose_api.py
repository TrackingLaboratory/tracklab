import os
import cv2
import torch
import requests
import numpy as np
from tqdm import tqdm
from functools import partial

import mmcv
from mmpose.apis import init_pose_model
from mmcv.parallel import collate, scatter
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmpose.core.post_processing import oks_nms


from pbtrack.datastruct import ImageMetadata, Detections, ImageMetadatas, Detection
from pbtrack import SingleDetector

import logging

log = logging.getLogger(__name__)
mmcv.collect_env()


def mmpose_collate(batch):
    return collate(batch, len(batch))


class TopDownMMPose(SingleDetector):
    collate_fn = mmpose_collate

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        self.check_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.model = init_pose_model(cfg.path_to_config, cfg.path_to_checkpoint, device)

        cfg = self.model.cfg
        self.dataset_info = DatasetInfo(cfg.dataset_info)
        self.test_pipeline = Compose(cfg.test_pipeline)

        self.flip_pairs = self.dataset_info.flip_pairs
        self.collate_fn = partial(self.collate_fn, samples_per_gpu=32)

    @torch.no_grad()
    def preprocess(self, detection: Detection, metadata: ImageMetadata):
        cfg = self.model.cfg
        image = cv2.imread(metadata.file_path)  # BGR not RGB !
        data = {
            "bbox": detection.bbox_ltwh,
            "img": image,
            "bbox_score": detection.bbox_score,
            "bbox_id": 0,
            "dataset": self.dataset_info.dataset_name,
            "joints_3d": np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            "joints_3d_visible": np.zeros(
                (cfg.data_cfg.num_joints, 3), dtype=np.float32
            ),
            "rotation": 0,
            "ann_info": {
                "image_size": np.array(cfg.data_cfg["image_size"]),
                "num_joints": cfg.data_cfg.num_joints,
                "flip_pairs": self.flip_pairs,
            },
        }
        return self.test_pipeline(data)

    @torch.no_grad()
    def process(self, batch, detections: Detections, metadatas: ImageMetadatas):
        batch = scatter(batch, [self.device])[0]
        keypoints_xycs = list(
            self.model(
                img=batch["img"],
                img_metas=batch["img_metas"],
                return_loss=False,
                return_heatmap=False,
            )["preds"]
        )
        keypoints_scores = []
        for keypoints_xyc, img_metas in zip(keypoints_xycs, batch["img_metas"]):
            visible_keypoints = keypoints_xyc[
                keypoints_xyc[:, 2] > self.cfg.min_keypoints_confidence
            ]
            keypoints_scores.append(
                np.mean(visible_keypoints[:, 2]) * img_metas["bbox_score"]
                if visible_keypoints.size > 0
                else 0.0
            )
        detections["keypoints_xyc"] = keypoints_xycs
        detections["keypoints_score"] = keypoints_scores
        detections.drop(
            detections[detections.keypoints_score < self.cfg.min_keypoints_score].index,
            inplace=True,
        )
        return detections

    def postprocess(self, detections: Detections, metadata: ImageMetadata = None):
        # FIXME do we want it here ? or in the tracker ?
        # we must be as lax as possible here and be severe in the tracker
        img_kpts = []
        for _, detection in detections.iterrows():
            img_kpts.append(
                {
                    "keypoints": detection.keypoints_xyc,
                    "score": detections.score,
                    "area": detections.bbox_ltwh[2] * detections.bbox_ltwh[3],
                }
            )
        keep = oks_nms(
            img_kpts,
            self.cfg.oks_threshold,
            vis_thr=self.cfg.visibility_threshold,
            sigmas=self.dataset_info.sigmas,
        )
        detections = detections.iloc[keep]  # FIXME won't work
        return detections

    @staticmethod
    def sanitize_bbox(bbox, image_shape):
        # FIXME do we want this ?
        # convert ltwh bbox to ltrb,check if bbox coordinates are within
        # image dimensions and round them to int
        new_l = np.clip(np.round(bbox[0]), 0, image_shape[0])
        new_t = np.clip(np.round(bbox[1]), 0, image_shape[1])
        new_r = np.clip(np.round(bbox[2] + new_l), 0, image_shape[0])
        new_b = np.clip(np.round(bbox[3] + new_t), 0, image_shape[1])
        return np.array([new_l, new_t, new_r, new_b], dtype=int)

    def sanitize_keypoints(self, keypoints, shape):
        # FIXME do we want this ?
        # Clip the keypoints to the image dimensions
        # and set the confidence to 0 if thresholds are not met
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, shape[0] - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, shape[1] - 1)
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
            progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True)
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
