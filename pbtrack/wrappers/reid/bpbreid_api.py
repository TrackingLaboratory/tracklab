import sys
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from yacs.config import CfgNode as CN

from pbtrack.utils.images import cv2_load_image
from .bpbreid_dataset import ReidDataset
from pbtrack.core.datastruct import ImageMetadata, Detection
from pbtrack.core.reidentifier import ReIdentifier
from pbtrack.utils.coordinates import kp_img_to_kp_bbox, rescale_keypoints
from plugins.reid.bpbreid.scripts.main import build_config, build_torchreid_model_engine
from plugins.reid.bpbreid.tools.feature_extractor import FeatureExtractor
from plugins.reid.bpbreid.torchreid.utils.imagetools import build_gaussian_heatmaps
from pbtrack.utils.collate import Unbatchable

from hydra.utils import to_absolute_path

sys.path.append(to_absolute_path("plugins/reid/bpbreid"))  # FIXME ugly
sys.path.append(to_absolute_path("plugins/reid"))  # FIXME ugly
import torchreid
from plugins.reid.bpbreid.torchreid.utils.tools import extract_test_embeddings
from plugins.reid.bpbreid.torchreid.data.masks_transforms import CocoToSixBodyMasks
from torchreid.data.datasets import configure_dataset_class

# need that line to not break import of torchreid ('from torchreid... import ...') inside the bpbreid.torchreid module
# to remove the 'from torchreid... import ...' error 'Unresolved reference 'torchreid' in PyCharm, right click
# on 'bpbreid' folder, then choose 'Mark Directory as' -> 'Sources root'
from bpbreid.scripts.default_config import engine_run_kwargs


class BPBReId(ReIdentifier):
    """
    TODO:
        why bbox move after strongsort?
        training
        batch process
        save config + commit hash with model weights
        model download from URL: HRNet etc
        save folder: uniform with reconnaissance
        wandb support
    """

    def __init__(
        self, cfg, tracking_dataset, dataset, device, save_path, model_pose, job_id
    ):
        tracking_dataset.name = dataset.name
        tracking_dataset.nickname = dataset.nickname
        additional_args = {
            "tracking_dataset": tracking_dataset,
            "reid_config": dataset,
            "pose_model": model_pose,
        }
        torchreid.data.register_image_dataset(
            tracking_dataset.name,
            configure_dataset_class(ReidDataset, **additional_args),
            tracking_dataset.nickname,
        )
        self.device = device
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg = build_config(config_file=self.cfg)
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = not self.cfg.test.evaluate
        self.feature_extractor = None
        self.model = None
        self.transform = CocoToSixBodyMasks()

    def preprocess(
        self, detection: Detection, metadata: ImageMetadata
    ):  # Tensor RGB (1, 3, H, W)
        mask_w, mask_h = 32, 64
        # image = metadata.load_image()  # FIXME load_image() doesn't work because of ImageMetadata.__getattr__(
        image = cv2_load_image(metadata.file_path)
        l, t, r, b = detection.bbox_ltrb.astype(int)
        crop = image[t:b, l:r]
        keypoints = detection.keypoints_xyc
        bbox_ltwh = np.array([l, t, r - l, b - t])
        kp_xyc_bbox = kp_img_to_kp_bbox(keypoints, bbox_ltwh)
        kp_xyc_mask = rescale_keypoints(
            kp_xyc_bbox, (bbox_ltwh[2], bbox_ltwh[3]), (mask_w, mask_h)
        )
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
        if not self.cfg.model.bpbreid.learnable_attention_enabled:
            pixels_parts_probabilities = build_gaussian_heatmaps(
                kp_xyc_mask, mask_w, mask_h
            )
            batch["masks"] = pixels_parts_probabilities

        # pixels_parts_probabilities = pixels_parts_probabilities[np.newaxis, ...]
        return batch

    def process(self, batch, detections, metadatas):
        im_crops = batch["img"]
        im_crops = [im_crop.numpy() for im_crop in im_crops]
        if "masks" in batch:
            external_parts_masks = batch["masks"]
            external_parts_masks = external_parts_masks.numpy()
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
        reid_df = pd.DataFrame(
            {
                "embeddings": list(embeddings),
                "visibility_scores": list(visibility_scores),
                "body_masks": list(body_masks),
            },
            index=detections.index,
        )
        detections = detections.merge(
            reid_df, left_index=True, right_index=True, validate="one_to_one"
        )
        return detections

    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))

    def _xywh_to_xyxy(self, bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2
