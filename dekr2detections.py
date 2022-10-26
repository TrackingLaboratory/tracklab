from types import SimpleNamespace
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from detections import Detections

from DEKR.lib.config import cfg
from DEKR.lib.config import update_config

import DEKR.tools._init_paths
import models
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.rescore import rescore_valid


class DEKR2detections():
    def __init__(self, p_cfg, device, vis_threshold=0.3):
        args = SimpleNamespace(cfg=p_cfg, opts=[])
        update_config(cfg, args)
        self.cfg = cfg
        self.device = device
        
        self.model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            self.cfg, is_train=False
        )
        if cfg.TEST.MODEL_FILE:
            self.model.load_state_dict(
                torch.load(cfg.TEST.MODEL_FILE, map_location=self.device), 
                strict=True
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # taken from DEKR/tools/inference_demo.py
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( # TODO FIXME (values to adapt I don't know how)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.max_img_size = cfg.DATASET.INPUT_SIZE
        self.vis_threshold =vis_threshold
        
    def image2input(self, image):
        assert 1 == image.shape[0], "Test batch size should be 1"
        x = image[0].cpu().numpy() # (1, 3, H, W) -> (3, H, W)
        input = np.transpose(x, (1, 2, 0)) # -> (H, W, 3)
        H, W = input.shape[:2]
        if W > H:
            ratio = float(self.max_img_size)/float(W)
            dim = (self.max_img_size, int(ratio*H))
            input = cv2.resize(input, dim, interpolation=cv2.INTER_LINEAR) # -> (h, w, 3)
        else:
            ratio = float(self.max_img_size)/float(H)
            dim = (int(ratio*W), self.max_img_size)
            input = cv2.resize(input, dim, interpolation=cv2.INTER_LINEAR) # -> (h, w, 3)
        return input
        
    @torch.no_grad()
    def estimate(self, image):
        assert isinstance(image, np.ndarray), f"Image must be a np.array, here: {type(image)}"
        assert image.ndim == 3, f"Image must be of dim = 3, here: {image.ndim}"
        assert image.shape[2] == 3, f"Image dims must be (h, w, 3), here: {image.shape}"
        assert image.shape[0] <= self.max_img_size and image.shape[1] <= self.max_img_size,\
            f"Image must be resized lower than {self.max_img_size}, here: {image.shape}"
        
        
        # make inference and extract results
        base_size, center, scale = get_multi_scale_size(
            image, self.cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )
        heatmap_sum = 0
        poses = []

        for scale in sorted(self.cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, self.cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = self.transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).to(self.device)

            heatmap, posemap = get_multi_stage_outputs(
                self.cfg, self.model, image_resized, self.cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                self.cfg, heatmap_sum, poses, heatmap, posemap, scale
            )
        
        heatmap_avg = heatmap_sum/len(self.cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(self.cfg, heatmap_avg, poses)
        
        results_poses = []
        results_scores = []
        if scores:
            if self.cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(self.cfg, poses, heatmap_avg)

            poses = get_final_preds(
                poses, center, scale_resized, base_size
            )
            if self.cfg.RESCORE.VALID:
                scores = rescore_valid(self.cfg, poses, scores)
                
            for score, pose in zip(scores, poses):
                if score >= self.vis_threshold:
                    results_scores.append(score)
                    results_poses.append(pose)            
            
        # converts results to detection
        detections = self._results2detections(image,
                                              results_poses,
                                              results_scores)
        return detections
        
    def _results2detections(self, image, results_poses, results_scores):
        height, width = image.shape[:2]
        detections = Detections(
            poses=results_poses,
            scores=results_scores,
            h=height,
            w=width
        )
        detections.update_bboxes()
        return detections
        

if __name__ == '__main__': # testing function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DEKR2detections("DEKR/experiments/inference.yaml", device)    
    
    from datasets import ImageFolder
    dataset = ImageFolder("../Yolov5_StrongSORT_OSNet/data/test_images")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    for i, image in enumerate(dataloader):
        input2detect = model.image2input(image)
        detections = model.estimate(input2detect)