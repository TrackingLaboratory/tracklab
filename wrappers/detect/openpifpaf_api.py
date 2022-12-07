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
        
    def pre_process(self, image):
        return image
    
    def process(self, pre_processed_batch):
        detections = []
        predictions, _, _ = self.predictor.image(pre_processed_batch.file_path)
        for prediction in predictions:
            detections.append(
                Detection(
                    image_id = pre_processed_batch.id,
                    video_id = pre_processed_batch.video_id,
                    keypoints_xyc = prediction.data,
                    bbox = kp_to_bbox_w_threshold(prediction.data, vis_threshold=0.05),
                    )  # type: ignore
                )
        return detections
