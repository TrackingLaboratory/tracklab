import torch
from torchvision.transforms.functional import to_pil_image

from pbtrack.core.detector import Detector
from pbtrack.datastruct.detections import Detection
from pbtrack.utils.coordinates import kp_to_bbox_w_threshold

import sys
from hydra.utils import to_absolute_path
sys.path.append(to_absolute_path("plugins/detect/openpifpaf/src"))
import openpifpaf

class OpenPifPaf(Detector):
    
    def __init__(self, cfg, device):
        assert cfg.checkpoint or cfg.train, "Either a checkpoint or train must be declared"
        self.cfg = cfg
        self.device = device
        
        if cfg.checkpoint:
            self.predictor = openpifpaf.Predictor(cfg.checkpoint)
            self.predictor.model.to(device)
    
    def train(self):
        saved_argv = sys.argv
        sys.argv += [f'--{str(k)}={str(v)}' for k, v in self.cfg.train.items()]
        self.cfg.checkpoint = openpifpaf.train.main()
        sys.argv = saved_argv
        self.predictor = openpifpaf.Predictor(self.cfg.checkpoint)
        self.predictor.model.to(self.device)
        
    def pre_process(self, image: torch.Tensor):
        # maybe check if the image should be by batch of alone
        # image = image[0]
        return to_pil_image(image)
    
    def process(self, image):
        return self.predictor.pil_image(image)
    
    def post_process(self, results, metadata):
        results, _, _ = results
        detections = []
        for result in results:
            result = result.data
            detections.append(
                Detection(
                    image_id = metadata.id,
                    video_id = metadata.video_id,
                    keypoints_xyc = result,
                    bbox = kp_to_bbox_w_threshold(result, vis_threshold=0.1),
                    )  # type: ignore
                )
        return detections
