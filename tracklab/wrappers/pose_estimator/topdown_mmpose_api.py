from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from mim import get_model_info
from mim.utils import get_installed_path
from mmpose.apis.inference import dataset_meta_from_config
from tqdm import tqdm

import mmcv
from mmpose.apis import init_model
from mmengine.dataset import Compose, default_collate

from tracklab.pipeline import ImageLevelModule, DetectionLevelModule
from tracklab.utils.openmmlab import get_checkpoint

import logging

log = logging.getLogger(__name__)


# def mmpose_collate(batch):
#     return collate(batch, len(batch))


class TopDownMMPose(DetectionLevelModule):
    collate_fn = default_collate
    input_columns = ["bbox_ltwh", "bbox_conf"]
    output_columns = ["keypoints_xyc", "keypoints_conf"]

    def __init__(self, device, batch_size, config_name, path_to_checkpoint,
                 vis_kp_threshold=0.4, min_num_vis_kp=3, **kwargs):
        super().__init__(batch_size)
        model_df = get_model_info(package="mmpose", configs=[config_name])
        if len(model_df) != 1:
            raise ValueError("Multiple values found for the config name")
        download_url = model_df.weight.item()
        package_path = Path(get_installed_path("mmpose"))
        path_to_config = package_path / ".mim" / model_df.config.item()
        get_checkpoint(path_to_checkpoint, download_url)
        self.model = init_model(str(path_to_config), path_to_checkpoint, device)
        self.vis_kp_threshold = vis_kp_threshold
        self.min_num_vis_kp = min_num_vis_kp
        self.dataset_info = dataset_meta_from_config(self.model.cfg, "test")

        # self.dataset_info = DatasetInfo(self.model.cfg.dataset_info)
        self.test_pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        data_info = dict(img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        data_info["bbox"] = detection.bbox.ltrb()[None]
        data_info["bbox_score"] = np.array(detection.bbox_conf)[None]
        data_info.update(self.model.dataset_meta)

        return self.test_pipeline(data_info)

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        results = self.model.test_step(batch)
        kps_xyc = []
        kps_conf = []
        for result in results:
            result = result.pred_instances
            keypoints = result.keypoints[0]
            visibility_scores = result.keypoints_visible[0]
            visibility_scores[visibility_scores < self.vis_kp_threshold] = 0
            keypoints_xyc = np.concatenate([keypoints, visibility_scores[:, None]], axis=-1)
            if len(np.nonzero(visibility_scores)[0]) < self.min_num_vis_kp:
                conf = 0
            else:
                conf = np.mean(visibility_scores[visibility_scores != 0])
            kps_xyc.append(keypoints_xyc)
            kps_conf.append(conf)
        detections["keypoints_conf"] = kps_conf
        detections["keypoints_xyc"] = kps_xyc
        return detections

