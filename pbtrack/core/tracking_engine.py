# TODO VICTOR
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader


from pbtrack.core import Detector, ReIdentifier, Tracker, EngineDatapipe
from pbtrack.datastruct.detections import Detections
from pbtrack.datastruct.image_metadatas import ImageMetadatas
from pbtrack.utils.collate import default_collate

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
    def __init__(self, detector: Detector, reider: ReIdentifier, tracker: Tracker, metadatas: ImageMetadatas):
        super().__init__()
        self.detector = detector
        self.reider = reider
        self.tracker = tracker
        self.metadatas = metadatas
    
    def predict_step(self, batch, batch_idx):
        """ Steps through tracking predictions for one image.

            This doesn't work for a batch or any other construct. To work on a batch
            we should modify the test_step to take a batch of images.

            Args:
                image (torch.Tensor): the image to process
                metadata (Images): metadata of the corresponding image       
            Returns:
                detection: populated detection object, with pose, reid and tracking info
        """
        idxs, batch = batch
        # 1. Detection

        detections = Detections(self.detector.process(batch, self.metadatas.loc[idxs]))

        # 2. Reid
        reid_detections = []
        reid_pipe = EngineDatapipe(self.reider, self.metadatas, detections)
        reid_dl = DataLoader(dataset=reid_pipe, batch_size=8, collate_fn=default_collate)
        for idxs, reid_batch in reid_dl:
            batch_detections = detections.iloc[idxs]
            batch_metadatas = self.metadatas.loc[batch_detections.image_id]
            reid_detections.append(self.reider.process(reid_batch, batch_detections, batch_metadatas))
            # reid_detections.append(self.reider.postprocess(reid_output))
        
        reid_detections = pd.concat(reid_detections)

        # 3. Tracking
        track_detections = []
        for detection in reid_detections:
            track_input = self.tracker.preprocess(detection)
            track_output = self.tracker(track_input)
            track_detections.append(self.tracker.postprocess(track_output))
        
        return pd.concat(track_detections)