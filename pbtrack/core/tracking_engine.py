# TODO VICTOR
import pytorch_lightning as pl
import pandas as pd
import torch

from pbtrack.datastruct.images import ImagesSeries

class OnlineTrackingEngine(pl.LightningModule):
    """ Online tracking engine
    
        Call with ::
            engine = OnlineTrackingEngine(detector, reider, tracker)
            trainer = pl.Trainer() # configure trainer
            detections = trainer.predict(engine, pose_dataloader)
            detections = pd.concat(detections)
        
        Args:
            detector: Predicts pose detection 
            reidentifier: Predicts reid embeddings
            tracker: Tracks using reid embeddings
    """
    def __init__(self, detector, reider, tracker):
        self.detector = detector
        self.reider = reider
        self.tracker = tracker
    
    def predict_step(self, image: torch.Tensor, Image: ImagesSeries):
        """ Steps through tracking predictions for one image.

            This doesn't work for a batch or any other construct. To work on a batch
            we should modify the test_step to take a batch of images.

            Args:
                image (torch.Tensor): the image to process
                Image (ImagesSeries): metadata of the corresponding image       
            Returns:
                detection: populated detection object, with pose, reid and tracking info
        """

        # 1. Detection
        pose_detections = self.detector(image, Image)

        # 2. Reid
        reid_detections = []
        for detection in pose_detections:
            reid_input = self.reider.preprocess(detection)
            reid_output = self.reider(reid_input)
            reid_detections.append(self.reider.postprocess(reid_output))
        
        reid_detections = pd.concat(reid_detections)

        # 3. Tracking
        track_detections = []
        for detection in reid_detections:
            track_input = self.tracker.preprocess(detection)
            track_output = self.tracker(track_input)
            track_detections.append(self.tracker.postprocess(track_output))
        
        return pd.concat(track_detections)