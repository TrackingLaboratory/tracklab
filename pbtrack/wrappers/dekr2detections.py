import sys
import cv2
import numpy as np
from types import SimpleNamespace
import torch
import torchvision.transforms as T

from pbtrack.tracker.tracker import Detection, Metadata, Keypoint, Bbox
import warnings

warnings.filterwarnings("ignore")
from modules.detect.DEKR.lib.config import cfg
from modules.detect.DEKR.lib.config import update_config
from hydra.utils import to_absolute_path

sys.path.append(to_absolute_path("modules/detect/DEKR/lib"))
import models
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.rescore import rescore_valid


@torch.no_grad()  # FIXME Is this required HERE?
class DEKR2detections:
    def __init__(self, cfg, device, vis_threshold=0.3):
        self.cfg = cfg
        self.device = device

        self.model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(
            self.cfg, is_train=False
        )
        if cfg.TEST.MODEL_FILE:
            self.model.load_state_dict(
                torch.load(cfg.TEST.MODEL_FILE, map_location=self.device), strict=True
            )

        self.model.to(self.device)
        self.model.eval()

        # taken from DEKR/tools/inference_demo.py
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(  # FIXME (values to adapt I don't know how)
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.max_img_size = cfg.DATASET.INPUT_SIZE
        self.vis_threshold = vis_threshold

    def _image2input(self, image):  # RGB| float 0 -> 1| tensor| (3, H, W)
        input = image.detach().cpu().numpy()  # tensor -> numpy
        input = np.transpose(input, (1, 2, 0))  # -> (H, W, 3)
        H, W = input.shape[:2]
        if W > H:
            ratio = float(self.max_img_size) / float(W)
            dim = (self.max_img_size, int(ratio * H))
            input = cv2.resize(
                input, dim, interpolation=cv2.INTER_LINEAR
            )  # -> (h, w, 3)
        else:
            ratio = float(self.max_img_size) / float(H)
            dim = (int(ratio * W), self.max_img_size)
            input = cv2.resize(
                input, dim, interpolation=cv2.INTER_LINEAR
            )  # -> (h, w, 3)
        return input

    @torch.no_grad()
    def run(self, data):
        input = self._image2input(data["image"])

        # make inference and extract results
        base_size, center, scale = get_multi_scale_size(
            input, self.cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )
        heatmap_sum = 0
        poses = []

        for scale in sorted(self.cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                input, self.cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = self.transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).to(self.device)

            heatmap, posemap = get_multi_stage_outputs(
                self.cfg, self.model, image_resized, self.cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                self.cfg, heatmap_sum, poses, heatmap, posemap, scale
            )

        heatmap_avg = heatmap_sum / len(self.cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(self.cfg, heatmap_avg, poses)

        results_poses = []
        results_scores = []
        if scores:
            if self.cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(self.cfg, poses, heatmap_avg)

            poses = get_final_preds(poses, center, scale_resized, base_size)
            if self.cfg.RESCORE.VALID:
                scores = rescore_valid(self.cfg, poses, scores)

            for score, pose in zip(scores, poses):
                if score >= self.vis_threshold:
                    results_scores.append(score)
                    results_poses.append(pose)
        results_scores = np.asarray(results_scores)
        results_poses = np.asarray(results_poses)
        h, w = input.shape[:2]
        # converts results to detection
        detection = self._results2detections(results_poses, results_scores, data, h, w)
        return detection, heatmap_avg

    def _results2detections(self, results_poses, results_scores, data, h, w):
        detections = []
        for score, pose in zip(results_scores, results_poses):
            detection = Detection()
            detection.metadata = Metadata(
                **{k: v for k, v in data.items() if k != "image"}
            )
            pose[:, :2] = detection.rescale_xy(pose[:, :2], (h, w))
            keypoints = []
            for part, keypoint in enumerate(pose):
                keypoints.append(Keypoint(keypoint[0], keypoint[1], keypoint[2], part))
            left_top = np.amin(pose, axis=0)
            bottom_right = np.amax(pose, axis=0)
            width = bottom_right[0] - left_top[0]
            height = bottom_right[1] - left_top[1]
            detection.keypoints = keypoints
            detection.bbox = Bbox(left_top[0], left_top[1], width, height, score)
            detection.source = 1
            detections.append(detection)

        if not detections:
            detection = Detection()
            detection.metadata = Metadata(
                **{k: v for k, v in data.items() if k != "image"}
            )
            detection.source = 0
            detections.append(detection)

        return detections
