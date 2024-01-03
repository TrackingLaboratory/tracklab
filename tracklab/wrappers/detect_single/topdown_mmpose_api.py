from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from mim import get_model_info
from mim.utils import get_installed_path
from tqdm import tqdm

import mmcv
from mmcv.parallel import collate, scatter
from mmpose.apis import init_pose_model
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.openmmlab import get_checkpoint

import logging

log = logging.getLogger(__name__)
mmcv.collect_env()


def mmpose_collate(batch):
    return collate(batch, len(batch))


class TopDownMMPose(ImageLevelModule):
    collate_fn = mmpose_collate
    output_columns = ["keypoints_xyc", "keypoints_conf"]

    def __init__(self, device, batch_size, config_name, path_to_checkpoint,
                 vis_kp_threshold=0.4, min_num_vis_kp=3):
        super().__init__(device, batch_size)
        model_df = get_model_info(package="mmpose", configs=[config_name])
        if len(model_df) != 1:
            raise ValueError("Multiple values found for the config name")
        download_url = model_df.weight.item()
        package_path = Path(get_installed_path("mmpose"))
        path_to_config = package_path / ".mim" / model_df.config.item()
        get_checkpoint(path_to_checkpoint, download_url)
        self.model = init_pose_model(str(path_to_config), path_to_checkpoint, device)
        self.vis_kp_threshold = vis_kp_threshold
        self.min_num_vis_kp = min_num_vis_kp
        self.dataset_info = DatasetInfo(self.model.cfg.dataset_info)
        self.test_pipeline = Compose(self.model.cfg.test_pipeline)

    @torch.no_grad()
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

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame):
        batch = scatter(batch, [self.device])[0]
        keypoints = self.model(
                img=batch["img"],
                img_metas=batch["img_metas"],
                return_loss=False,
                return_heatmap=False,
            )["preds"]
        keypoints[keypoints[:, :, 2] < self.vis_kp_threshold, 2] = 0
        keypoints = list(keypoints)
        confs = []
        for kp in keypoints:
            if kp[kp[:, 2] != 0].shape[0] < self.min_num_vis_kp:
                confs.append(0.)
            else:
                confs.append(np.mean(kp[kp[:, 2] != 0, 2]))
        detections["keypoints_conf"] = confs
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
