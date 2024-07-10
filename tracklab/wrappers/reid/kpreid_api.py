import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from yacs.config import CfgNode as CN

import kpreid
from kpreid.data import ImageDataset
from kpreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from kpreid.data.masks_transforms import masks_preprocess_all
from kpreid.data.transforms import build_transforms
from kpreid.tools.extract_part_based_features import extract_reid_features
from .bpbreid_dataset import ReidDataset
# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.coordinates import rescale_keypoints, clip_keypoints_to_image
from tracklab.utils.collate import default_collate

from kpreid.scripts.main import build_config, build_torchreid_model_engine, build_model

from pathlib import Path

from kpreid.utils.tools import extract_test_embeddings
from kpreid.data.datasets import configure_dataset_class

from kpreid.scripts.default_config import engine_run_kwargs

from ...pipeline.detectionlevel_module import DetectionLevelModule
from ...utils.download import download_file


class KPReId(DetectionLevelModule):
    """
    """

    collate_fn = default_collate
    input_columns = ["bbox_ltwh", "bbox_conf", "keypoints_xyc", "negative_kps"]
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
        kpreid.data.register_image_dataset(
            tracking_dataset.name,
            configure_dataset_class(ReidDataset, **additional_args),
            tracking_dataset.nickname,
        )
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        # self.download_models(load_weights=self.cfg.model.load_weights,
        #                      pretrained_path=self.cfg.model.bpbreid.hrnet_pretrained_path,
        #                      backbone=self.cfg.model.bpbreid.backbone)
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config_file=self.cfg)
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = training_enabled
        self.feature_extractor = None
        self.model = None
        self.coco_transform = masks_preprocess_all[self.cfg.model.bpbreid.masks.preprocess]() if \
            self.cfg.model.bpbreid.masks.preprocess != 'none' else None
        self.keypoints_to_prompt_masks = KeypointsToMasks(mode=self.cfg.model.bpbreid.keypoints.prompt_masks,
                                                          vis_thresh=self.cfg.model.bpbreid.keypoints.vis_thresh,
                                                          vis_continous=self.cfg.model.bpbreid.keypoints.vis_continous,
                                                          )
        self.keypoints_to_target_masks = KeypointsToMasks(
            mode=self.cfg.model.bpbreid.keypoints.target_masks,
            vis_thresh=self.cfg.model.bpbreid.keypoints.vis_thresh,
            vis_continous=False,
            )
        self.model = build_model(self.cfg, 0)
        self.model.eval()

        _, self.transforms, self.target_preprocess, self.prompt_preprocess = build_transforms(
            self.cfg.data.height,
            self.cfg.data.width,
            self.cfg,
            transforms=self.cfg.data.transforms,
            norm_mean=self.cfg.data.norm_mean,
            norm_std=self.cfg.data.norm_std,
            remove_background_mask=False,
            masks_preprocess=self.cfg.model.bpbreid.masks.preprocess,
            softmax_weight=self.cfg.model.bpbreid.masks.softmax_weight,
            background_computation_strategy=self.cfg.model.bpbreid.masks.background_computation_strategy,
            mask_filtering_threshold=self.cfg.model.bpbreid.masks.mask_filtering_threshold,
            train_dir=None,
        )

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

        sample = {
            "image": crop,
            "keypoints_xyc": clip_keypoints_to_image(detection.keypoints.keypoints_bbox_xyc(), (crop.shape[1]-1, crop.shape[0]-1)),
            "negative_kps": clip_keypoints_to_image(detection.negative_kps, (crop.shape[1]-1, crop.shape[0]-1)),
        }

        batch = ImageDataset.getitem(
            sample,
            self.cfg,
            self.keypoints_to_prompt_masks,
            self.prompt_preprocess,
            self.keypoints_to_target_masks,
            self.target_preprocess,
            self.transforms,
            load_masks=True,
        )

        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        args = {}
        args["images"] = batch["image"]
        if "prompt_masks" in batch:
            args["prompt_masks"] = batch["prompt_masks"]
        model_output = self.model(**args)

        (
            embeddings,
            visibility_scores,
            parts_masks,
            pixels_cls_scores,
        ) = extract_test_embeddings(
            model_output, self.cfg.model.bpbreid.test_embeddings
        )

        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()
        parts_masks = parts_masks.cpu().detach().numpy()

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
                "body_masks": list(parts_masks),
            },
            index=detections.index,
        )
        return reid_df

    def train(self, *args, **kwargs):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        if not self.cfg.inference.enabled:
            self.engine.run(**engine_run_kwargs(self.cfg))
            self.model.eval()
        else:
            print("Starting inference on external data")
            extract_reid_features(self.cfg, self.cfg.inference.input_folder, self.cfg.inference.output_folder, self.cfg.inference.output_figure_folder, model=self.model)
