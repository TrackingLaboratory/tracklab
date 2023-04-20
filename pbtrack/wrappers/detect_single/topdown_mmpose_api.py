import cv2
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm

import mmcv
from mmcv.parallel import collate, scatter
from mmpose.apis import init_pose_model
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose

from pbtrack import SingleDetector
from pbtrack.utils.openmmlab import get_checkpoint

import logging

log = logging.getLogger(__name__)
mmcv.collect_env()


def mmpose_collate(batch):
    return collate(batch, len(batch))


@torch.no_grad()
class TopDownMMPose(SingleDetector):
    collate_fn = mmpose_collate

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        get_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.model = init_pose_model(cfg.path_to_config, cfg.path_to_checkpoint, device)

        self.dataset_info = DatasetInfo(self.model.cfg.dataset_info)
        self.test_pipeline = Compose(self.model.cfg.test_pipeline)

    def preprocess(self, detection: pd.Series, metadata: pd.Series):
        image = cv2.imread(metadata.file_path)  # BGR not RGB !
        data = {
            "bbox": detection.bbox_ltwh,
            "img": image,
            "bbox_score": detection.bbox_conf,
            "bbox_id": 0,
            "dataset": self.dataset_info.dataset_name,
            "joints_3d": np.zeros(
                (self.model.cfg.data_cfg.num_joints, 3), dtype=np.float32
            ),
            "joints_3d_visible": np.zeros(
                (self.model.cfg.data_cfg.num_joints, 3), dtype=np.float32
            ),
            "rotation": 0,
            "ann_info": {
                "image_size": np.array(self.model.cfg.data_cfg["image_size"]),
                "num_joints": self.model.cfg.data_cfg.num_joints,
                "flip_pairs": self.dataset_info.flip_pairs,
            },
        }
        return self.test_pipeline(data)

    def process(self, batch, detections: pd.DataFrame):
        batch = scatter(batch, [self.device])[0]
        keypoints = list(
            self.model(
                img=batch["img"],
                img_metas=batch["img_metas"],
                return_loss=False,
                return_heatmap=False,
            )["preds"]
        )
        detections["keypoints_conf"] = [
            np.mean(kp[:, 2]) * s["bbox_score"]
            for kp, s in zip(keypoints, batch["img_metas"])
        ]
        detections["keypoints_xyc"] = keypoints
        return detections

    """
    from mmpose.core.post_processing import oks_nms
    def postprocess(self, detections: pd.DataFrame, metadata: pd.DataFrame = None):
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
    """
