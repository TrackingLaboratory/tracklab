import os
import sys
import numpy as np
import torch
sys.path.append('bpbreid')
# need that line to not break import of torchreid ('from torchreid... import ...') inside the bpbreid.torchreid module
# to remove the 'from torchreid... import ...' error 'Unresolved reference 'torchreid' in PyCharm, right click
# on 'bpbreid' folder, then choose 'Mark Directory as' -> 'Sources root'
from bpbreid.scripts.default_config import get_default_config, display_config_diff
from torchreid.utils import FeatureExtractor


@torch.no_grad()
class Torchreid2detections():
    """
    TODO:
        why bbox move after strongsort?
        training
        batch process
        save config + commit hash with model weights
        model download from URL
    """
    def __init__(self, device, save_path):
        model_path = '/Users/vladimirsomers/Models/BPBReID/hrnet_jobid_8728_model.pth.tar-120'
        config_file_path = 'bpbreid/configs/bpbreid/remote_bpbreid_market1501_test.yaml'
        cfg = get_default_config()
        cfg.data.parts_num = 5
        cfg.use_gpu = torch.cuda.is_available()
        default_cfg_copy = cfg.clone()
        cfg.merge_from_file(config_file_path)
        cfg.project.config_file = os.path.basename(config_file_path)
        display_config_diff(cfg, default_cfg_copy)
        cfg.model.pretrained = False
        num_classes = 702  # for model trained on DukeMTMC
        cfg.data.save_dir = save_path
        self.model = FeatureExtractor(
            cfg,
            model_path=model_path,
            device=device,
            num_classes=num_classes
        )

    def _image2input(self, image): # Tensor RGB (1, 3, H, W)
        assert 1 == image.shape[0], "Test batch size should be 1"
        input = image[0].cpu().numpy() # -> (3, H, W)
        input = np.transpose(input, (1, 2, 0)) # -> (H, W, 3)
        input = input*255.0
        input = input.astype(np.uint8) # -> to uint8
        return input

    def run(self, detections, image):
        im_crops = []
        image = self._image2input(image)
        for det in detections.bboxes:
            box = det
            height, width = image.shape[:2]
            x1, y1, x2, y2 = self._xywh_to_xyxy(box, width, height)
            crop = image[y1:y2, x1:x2]
            im_crops.append(crop)
        if im_crops:
            embeddings, visibility_scores, body_masks, _ = self.model(im_crops)
            detections.add_reid_features(embeddings)
            detections.add_visibility_scores(visibility_scores)
            detections.add_body_masks(body_masks)
        return detections

    def _xywh_to_xyxy(self, bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2
