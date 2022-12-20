import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from pbtrack.core import Detector, ReIdentifier, Tracker, EngineDatapipe
from pbtrack.core.datastruct import Detections
from pbtrack.core.datastruct.image_metadatas import ImageMetadatas
from pbtrack.core.datastruct.tracker_state import TrackerState
from pbtrack.utils.collate import default_collate


class OnlineTrackingEngine(pl.LightningModule):
    """Online tracking engine

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

    def __init__(
        self,
        detector: Detector,
        reider: ReIdentifier,
        tracker: Tracker,
        tracker_state: TrackerState,
        vis_engine,
        detect_batchsize: int,
        reid_batchsize: int,
    ):
        super().__init__()
        self.trainer = pl.Trainer()
        self.vis_engine = vis_engine  # TODO : Convert to callback
        self.tracker_state = tracker_state
        self.detector = detector
        self.reider = reider
        self.tracker = tracker
        self.img_metadatas = tracker_state.gt.image_metadatas
        self.video_metadatas = tracker_state.gt.video_metadatas
        self.detect_batchsize = detect_batchsize
        self.reid_batchsize = reid_batchsize

    def run(self):
        for video_idx, video in self.video_metadatas.iterrows():
            self.video_step(video, video_idx)

    def video_step(self, video, video_id):
        """Run tracking on a single video.

        Updates the tracking state, and creates the visualization for the
        current video.

        Args:
            video: a VideoMetadata series
            video_id: the video id. must be the same as video.name

        """
        imgs_meta = self.img_metadatas
        detection_datapipe = DataLoader(
            dataset=EngineDatapipe(
                self.detector, imgs_meta[imgs_meta.video_id == video_id]
            ),
            batch_size=self.detect_batchsize,
        )
        self.tracker.reset()

        detections_list = self.trainer.predict(self, dataloaders=detection_datapipe)
        detections = pd.concat(detections_list)
        self.tracker_state.update(detections)
        self.vis_engine.run(self.tracker_state, video_id)
        self.tracker_state.free(video_id)

    def predict_step(self, batch, batch_idx):
        """Steps through tracking predictions for one image.

        This doesn't work for a batch or any other construct. To work on a batch
        we should modify the test_step to take a batch of images.

        Args:
            batch : a tuple containing the image indexes and the input as needed
                by the detector
            batch_idx : the batch index (currently unused)
        Returns:
            detection_list : list of detection objects, with pose, reid and tracking info
        """
        idxs, batch = batch
        image_metadatas = self.img_metadatas.loc[idxs]

        # 1. Detection
        detections = Detections(self.detector.process(batch, image_metadatas))
        if detections.empty:
            return detections

        # 2. Reid
        reid_detections = []
        reid_pipe = EngineDatapipe(self.reider, self.img_metadatas, detections)
        reid_dl = DataLoader(
            dataset=reid_pipe,
            batch_size=self.reid_batchsize,
            collate_fn=default_collate,
        )
        for idxs, reid_batch in reid_dl:
            batch_detections = detections.loc[idxs]
            batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
            reid_detections.append(
                self.reider.process(reid_batch, batch_detections, batch_metadatas)
            )
            # reid_detections.append(self.reider.postprocess(reid_output))

        if len(reid_detections) > 0:
            reid_detections = pd.concat(reid_detections)
        else:
            reid_detections = Detections()

        # 3. Tracking
        track_detections = []
        for image_id, image_metadata in image_metadatas.iterrows():
            if len(reid_detections) == 0:
                continue
            detections = reid_detections[reid_detections.image_id == image_id]
            if len(detections) == 0:
                continue
            track_pipe = EngineDatapipe(self.tracker, self.img_metadatas, detections)
            track_dl = DataLoader(dataset=track_pipe, batch_size=2**16)
            for idxs, track_batch in track_dl:
                batch_detections = detections.loc[idxs]
                batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
                track_detections.append(
                    self.tracker.process(track_batch, batch_detections, batch_metadatas)
                )

        if len(track_detections) > 0:
            return pd.concat(track_detections)
        else:
            return Detections()
