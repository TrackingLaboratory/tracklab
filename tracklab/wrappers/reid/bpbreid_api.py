import gdown
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from yacs.config import CfgNode as CN
from .bpbreid_dataset import ReidDataset
# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.coordinates import rescale_keypoints
from tracklab.utils.collate import default_collate

from torchreid.scripts.main import build_config, build_torchreid_model_engine
from torchreid.tools.feature_extractor import FeatureExtractor
from torchreid.utils.imagetools import (
    build_gaussian_heatmaps,
)
from tracklab.utils.collate import Unbatchable

import tracklab
from pathlib import Path


import torchreid
from torch.nn import functional as F
from torchreid.data.masks_transforms import (
    CocoToSixBodyMasks,
    masks_preprocess_transforms,
)
from torchreid.utils.tools import extract_test_embeddings
from torchreid.data.datasets import configure_dataset_class

from torchreid.scripts.default_config import engine_run_kwargs

from ...pipeline.detectionlevel_module import DetectionLevelModule
from ...utils.download import download_file


class BPBReId(DetectionLevelModule):
    """
    TODO:
        why bbox move after strong_sort?
        training
        batch process
        save config + commit hash with model weights
        model download from URL: HRNet etc
        save folder: uniform with reconnaissance
        wandb support
    """

    collate_fn = default_collate
    input_columns = ["bbox_ltwh"]
    output_columns = ["embeddings", "visibility_scores", "body_masks"]

    def __init__(
        self,
        cfg,
        tracking_dataset,
        dataset,
        device,
        save_path,
        job_id,
        use_keypoints_visibility_scores_for_reid,
        training_enabled,
        batch_size,
    ):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        tracking_dataset.name = dataset.name
        tracking_dataset.nickname = dataset.nickname
        self.dataset_cfg = dataset
        self.use_keypoints_visibility_scores_for_reid = (
            use_keypoints_visibility_scores_for_reid
        )
        tracking_dataset.name = self.dataset_cfg.name
        tracking_dataset.nickname = self.dataset_cfg.nickname
        additional_args = {
            "tracking_dataset": tracking_dataset,
            "reid_config": self.dataset_cfg,
            "pose_model": None,
        }
        torchreid.data.register_image_dataset(
            tracking_dataset.name,
            configure_dataset_class(ReidDataset, **additional_args),
            tracking_dataset.nickname,
        )
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        self.download_models(load_weights=self.cfg.model.load_weights,
                             pretrained_path=self.cfg.model.bpbreid.hrnet_pretrained_path,
                             backbone=self.cfg.model.bpbreid.backbone)
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config=self.cfg)
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = training_enabled
        self.feature_extractor = None
        self.model = None

    def download_models(self, load_weights, pretrained_path, backbone):
        if Path(load_weights).stem == "bpbreid_market1501_hrnet32_10642":
            md5 = "e79262f17e7486ece33eebe198c07841"
            download_file("https://zenodo.org/records/10604211/files/bpbreid_market1501_hrnet32_10642.pth?download=1",
                          local_filename=load_weights, md5=md5)
        if backbone == "hrnet32":
            md5 = "58ea12b0420aa3adaa2f74114c9f9721"
            path = Path(pretrained_path) / "hrnetv2_w32_imagenet_pretrained.pth"
            download_file("https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1",
                          local_filename=path, md5=md5)

    @torch.no_grad()
    def preprocess(
        self, image, detection: pd.Series, metadata: pd.Series
    ):  # Tensor RGB (1, 3, H, W)
        mask_w, mask_h = 32, 64
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
        if not self.cfg.model.bpbreid.learnable_attention_enabled:
            bbox_ltwh = detection.bbox.ltwh(
                image_shape=(image.shape[1], image.shape[0]), rounded=True
            )
            kp_xyc_bbox = detection.keypoints.in_bbox_coord(bbox_ltwh)
            kp_xyc_mask = rescale_keypoints(
                kp_xyc_bbox, (bbox_ltwh[2], bbox_ltwh[3]), (mask_w, mask_h)
            )
            if self.dataset_cfg.masks_mode == "gaussian_keypoints":
                pixels_parts_probabilities = build_gaussian_heatmaps(
                    kp_xyc_mask, mask_w, mask_h
                )
            else:
                raise NotImplementedError
            batch["masks"] = pixels_parts_probabilities

        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        im_crops = batch["img"]
        im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]
        if "masks" in batch:
            external_parts_masks = batch["masks"]
            external_parts_masks = external_parts_masks.cpu().detach().numpy()
        else:
            external_parts_masks = None
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.device,
                image_size=(self.cfg.data.height, self.cfg.data.width),
                model=self.model,
                verbose=False,  # FIXME @Vladimir
            )
        reid_result = self.feature_extractor(
            im_crops, external_parts_masks=external_parts_masks
        )
        embeddings, visibility_scores, body_masks, _ = extract_test_embeddings(
            reid_result, self.test_embeddings
        )

        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()
        body_masks = body_masks.cpu().detach().numpy()

        if self.use_keypoints_visibility_scores_for_reid:
            kp_visibility_scores = batch["visibility_scores"].numpy()
            if visibility_scores.shape[1] > kp_visibility_scores.shape[1]:
                kp_visibility_scores = np.concatenate(
                    [np.ones((visibility_scores.shape[0], 1)), kp_visibility_scores],
                    axis=1,
                )
            visibility_scores = np.float32(kp_visibility_scores)

        reid_df = pd.DataFrame(
            {
                "embeddings": list(embeddings),
                "visibility_scores": list(visibility_scores),
                "body_masks": list(body_masks),
            },
            index=detections.index,
        )
        return reid_df

    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))
