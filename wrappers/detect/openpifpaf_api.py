from pbtrack.core.detector import Detector
import sys
sys.path.append("plugins/detect/openpifpaf/src")
import openpifpaf

class OpenPifPaf(Detector):
    
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model
    
    """ TODO
    def train(self):
        saved_argv = sys.argv
        sys.argv += [f'--{str(k)}={str(v)}' for k, v in self.cfg.train.items()]
        checkpoint = openpifpaf.train.main()
        self.cfg.checkpoint = checkpoint
        sys.argv = saved_argv
        
    def run(self, tracker):
        predictor = openpifpaf.Predictor(self.cfg.checkpoint)
        
        images = tracker.get_images()
        predictions, gt_anns, image_meta = predictor.images(images)
        tracker.update(predictions)
    """
