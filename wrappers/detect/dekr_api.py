import sys
import cv2
import torch
import torchvision.transforms as T

import warnings

warnings.filterwarnings("ignore")

from pbtrack.datastruct.detections import Detection
from pbtrack.core.detector import Detector
from pbtrack.utils.coordinates import kp_to_bbox

from hydra.utils import to_absolute_path

sys.path.append(to_absolute_path("plugins/detect/DEKR/lib"))

import models
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.rescore import rescore_valid

# TODO add train
class DEKR(Detector):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.id = 0

        self.model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(  # FIXME uggly
            self.cfg, is_train=False
        )
        if cfg.TEST.MODEL_FILE:
            self.model.load_state_dict(
                torch.load(cfg.TEST.MODEL_FILE, map_location=self.device), strict=True
            )
        self.model.to(self.device)

        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(  # COCO specific
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image):
        # load images
        image = cv2.imread(
            str(image.file_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # -> (H, W, 3)
        # augmented images
        image_resizeds = []
        scales = []
        base_size, center, scale = get_multi_scale_size(
            image, self.cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )
        for scale in sorted(self.cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, self.cfg.DATASET.INPUT_SIZE, scale, 1.0
            )
            image_resized = self.transforms(image_resized)
            image_resizeds.append(image_resized)
            scales.append(scale)
        return {
            "base_size": torch.tensor(base_size, dtype=torch.int32),
            "center": torch.tensor(center, dtype=torch.int32),
            "scale_resized": torch.tensor(scale_resized),
            "scales": torch.tensor(scales),
            "image_resizeds": torch.stack(image_resizeds),
        }

    def _reshape_image(self, image):
        init_shape = image.shape[:2]
        if image.shape[0] > image.shape[1]:  # H > W
            ratio = float(self.cfg.DATASET.INPUT_SIZE) / float(image.shape[0])
            dim = (
                int(ratio * image.shape[1]),
                self.cfg.DATASET.INPUT_SIZE,
            )  # new width, height
        else:  # W > H
            ratio = float(self.cfg.DATASET.INPUT_SIZE) / float(image.shape[1])
            dim = (
                self.cfg.DATASET.INPUT_SIZE,
                int(ratio * image.shape[0]),
            )  # new width, height
        image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)  # -> (h, w, 3)
        new_shape = image.shape[:2]
        return image, torch.tensor(init_shape), torch.tensor(new_shape)

    def process(self, preprocessed_batch, metadatas):
        base_sizes = preprocessed_batch["base_size"]
        centers = preprocessed_batch["center"]
        scale_resizeds = preprocessed_batch["scale_resized"]
        scaless = preprocessed_batch["scales"]
        image_resizedss = preprocessed_batch["image_resizeds"]

        detections = []
        for (
            (_, metadata),
            base_size,
            center,
            scale_resized,
            scales,
            image_resizeds,
        ) in zip(
            metadatas.iterrows(),
            base_sizes,
            centers,
            scale_resizeds,
            scaless,
            image_resizedss,
        ):
            poses = self._process_images(
                image_resizeds, scales, base_size, center, scale_resized
            )
            for pose in poses:
                detections.append(
                    Detection.create(
                        image_id=metadata.id,
                        id=self.id,
                        bbox=kp_to_bbox(pose),
                        keypoints_xyc=pose,
                    )  # type: ignore
                )
                self.id += 1

        return detections

    @torch.no_grad()  # required
    def _process_images(self, image_resizeds, scales, base_size, center, scale_resized):
        heatmap_sum = 0
        poses = []
        for image_resized, scale in zip(image_resizeds, scales):
            image_resized = image_resized.unsqueeze(0).to(self.device)

            heatmap, posemap = get_multi_stage_outputs(
                self.cfg, self.model, image_resized, self.cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                self.cfg, heatmap_sum, poses, heatmap, posemap, scale
            )

        heatmap_avg = heatmap_sum / len(self.cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(self.cfg, heatmap_avg, poses)
        if scores:
            if self.cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(self.cfg, poses, heatmap_avg)
            poses = get_final_preds(
                poses, center.numpy(), scale_resized.numpy(), base_size.numpy()
            )
            if self.cfg.RESCORE.VALID:
                scores = rescore_valid(self.cfg, poses, scores)

        results_poses = []
        results_scores = []
        for pose, score in zip(poses, scores):
            if score >= self.cfg.TEST.visibility_thre:
                results_scores.append(score)
                results_poses.append(pose)
        return results_poses  # , results_scores
