import torch

from pbtrack.core.detector import Detector

import sys
sys.path.append("plugins/detect/openpifpaf/src")
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
        return image.cpu().detach().numpy()
    
    def process(self, image):
        return self.predictor.numpy_image(image)
    
    def post_process(self, results, Image):
        results, _, _ = results
        # TODO
        return 
