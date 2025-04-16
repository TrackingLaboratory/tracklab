from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torchreid.data import ImageDataset, masks_preprocess_all
from torchreid.data.datasets.image.occluded_posetrack21 import clip_keypoints_to_image
from torchreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from torchreid.data.transforms import build_transforms
from torchreid.scripts.builder import build_config, build_torchreid_model_engine, build_model
from torchreid.scripts.default_config import engine_run_kwargs
from torchreid.utils.tools import extract_test_embeddings
from yacs.config import CfgNode as CN

from tracklab.utils.collate import default_collate
from ...pipeline.detectionlevel_module import DetectionLevelModule


# TODO clean code
class KPReId(DetectionLevelModule):
    collate_fn = default_collate
    input_columns = ["bbox_ltwh"]
    output_columns = ["embeddings", "visibility_scores"]

    def __init__(
        self,
        cfg,
        device,
        save_path,
        training_enabled,
        batch_size,
        job_id=0,
        *args,
        **kwargs
    ):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device

        # registering Tracklab's datasets with Torchreid (only the ones required by Torchreid)
        all_configured_datasets = set(cfg.data.sources + cfg.data.targets)  # all datasets required by Torchreid
        #tracking_datasets = [d for _, d in datasets.items() if d.name in all_configured_datasets]
        #for tracking_dataset in tracking_datasets:
        #    self.dataset_cfg = dataset
        #    self.use_keypoints_visibility_scores_for_reid = use_keypoints_visibility_scores_for_reid
        #    additional_args = {
        #        "tracking_dataset": tracking_dataset,
        #        "reid_config": self.dataset_cfg,
        #    }
        #    print(f"Registering dataset {tracking_dataset.name}")
        #    TorchreidDataset = type(tracking_dataset.name, (ReidDataset,), {})  #
        #    register_image_dataset(
        #        tracking_dataset.name,
        #        configure_dataset_class(TorchreidDataset, **additional_args),
        #        tracking_dataset.nickname,
        #    )
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        self.download_models(load_weights=self.cfg.model.load_weights)
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config=self.cfg, training_enabled=training_enabled)
        self.test_embeddings = self.cfg.model.kpr.test_embeddings
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = training_enabled
        self.feature_extractor = None
        self.model = None

        self.coco_transform = masks_preprocess_all[self.cfg.model.kpr.masks.preprocess]() if \
            self.cfg.model.kpr.masks.preprocess \
        != 'none' else None

        self.keypoints_to_prompt_masks = KeypointsToMasks(mode=self.cfg.model.kpr.keypoints.prompt_masks,
                                                          vis_thresh=self.cfg.model.kpr.keypoints.vis_thresh,
                                                          vis_continous=self.cfg.model.kpr.keypoints.vis_continous,
                                                          )

        self.keypoints_to_target_masks = KeypointsToMasks(mode=self.cfg.model.kpr.keypoints.target_masks,
                                                          vis_thresh=self.cfg.model.kpr.keypoints.vis_thresh,
                                                          vis_continous=False,
                                                          )

        self.model = build_model(self.cfg, 0, verbose=False)
        self.model.eval()

        _, self.transforms, self.target_preprocess, self.prompt_preprocess = build_transforms(
            self.cfg.data.height,
            self.cfg.data.width,
            self.cfg,
            transforms=self.cfg.data.transforms,
            norm_mean=self.cfg.data.norm_mean,
            norm_std=self.cfg.data.norm_std,
            remove_background_mask=False,
            masks_preprocess=self.cfg.model.kpr.masks.preprocess,
            softmax_weight=self.cfg.model.kpr.masks.softmax_weight,
            background_computation_strategy=self.cfg.model.kpr.masks.background_computation_strategy,
            mask_filtering_threshold=self.cfg.model.kpr.masks.mask_filtering_threshold,
            train_dir=None,
            verbose=False,
        )

    def download_models(self, load_weights):
        load_weights = Path(load_weights)
        if not load_weights.is_file():
            hf_hub_download(
                repo_id="trackinglaboratory/keypoint_promptable_reid",
                filename=load_weights.name,
                local_dir=load_weights.parent
            )

    @torch.no_grad()
    def preprocess(
        self, image, detection: pd.Series, metadata: pd.Series
    ):  # Tensor RGB (1, 3, H, W)
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]

        sample = {
            "image": crop,
        }

        if "keypoints" in detection:
            sample["keypoints_xyc"] = clip_keypoints_to_image(detection.keypoints.keypoints_bbox_xyc(),
                                                     (crop.shape[1] - 1, crop.shape[0] - 1))
        if "negative_kps" in detection:
            sample["negative_kps"] = clip_keypoints_to_image(detection.negative_kps, (crop.shape[1] - 1, crop.shape[0] - 1))

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
            model_output, self.cfg.model.kpr.test_embeddings
        )

        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()

        #if self.use_keypoints_visibility_scores_for_reid:
        #    kp_visibility_scores = batch["visibility_scores"].numpy()
        #    if visibility_scores.shape[1] > kp_visibility_scores.shape[1]:
        #        kp_visibility_scores = np.concatenate(
        #            [np.ones((visibility_scores.shape[0], 1)), kp_visibility_scores],
        #            axis=1,
        #        )
        #    visibility_scores = np.float32(kp_visibility_scores)

        reid_df = pd.DataFrame(
            {
                "embeddings": list(embeddings),
                "visibility_scores": list(visibility_scores),
            },
            index=detections.index,
        )
        return reid_df

    def train(self, *args, **kwargs):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))