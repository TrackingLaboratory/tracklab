import pandas as pd
import pytorch_lightning as pl
import logging
import warnings

from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm import tqdm

from pbtrack.core import Detector, ReIdentifier, Tracker, EngineDatapipe
from pbtrack.core.datastruct import Detections
from pbtrack.core.datastruct.tracker_state import TrackerState
from pbtrack.utils.collate import default_collate
from pbtrack.utils.images import cv2_load_image

log = logging.getLogger(__name__)

import warnings

warnings.filterwarnings(
    "ignore", ".*does not have many workers.*"
)  # Disable UserWarning for DataLoaders with num_workers=0


class OnlineTrackingEngine(pl.LightningModule):
    """Online tracking engine

    Call with ::
        engine = OnlineTrackingEngine(model_detect, model_reid, model_track)
        trainer = pl.Trainer() # configure trainer
        detections = trainer.predict(engine, pose_dataloader)
        detections = pd.concat(detections)

    Args:
        model_detect: Predicts pose detection
        reidentifier: Predicts reid embeddings
        model_track: Tracks using reid embeddings
    """

    def __init__(
        self,
        model_detect: Detector,
        model_reid: ReIdentifier,
        model_track: Tracker,
        tracker_state: TrackerState,
        vis_engine,
        detect_batchsize: int,
        reid_batchsize: int,
        num_workers: int,
    ):
        super().__init__()
        self.trainer_model = pl.Trainer()
        self.trainer_track = pl.Trainer(enable_progress_bar=False)
        self.vis_engine = vis_engine  # TODO : Convert to callback
        self.tracker_state = tracker_state
        self.model_detect = model_detect
        self.model_reid = model_reid
        self.model_track = model_track
        self.img_metadatas = tracker_state.gt.image_metadatas
        self.video_metadatas = tracker_state.gt.video_metadatas
        self.detect_batchsize = detect_batchsize
        self.reid_batchsize = reid_batchsize
        self.detection_datapipe = EngineDatapipe(self.model_detect)
        self.detection_dl = DataLoader(
            dataset=self.detection_datapipe,
            batch_size=self.detect_batchsize,
            collate_fn=type(self.model_detect).collate_fn,
            num_workers=num_workers,
            persistent_workers=False,
        )
        self.reid_datapipe = EngineDatapipe(self.model_reid)
        self.reid_dl = DataLoader(
            dataset=self.reid_datapipe,
            batch_size=self.reid_batchsize,
            num_workers=num_workers,
            collate_fn=default_collate,
            persistent_workers=False,
        )
        self.track_datapipe = EngineDatapipe(self.model_track)
        self.track_dl = DataLoader(
            dataset=self.track_datapipe,
            batch_size=2**16,
            num_workers=0,
            persistent_workers=False,
        )

    def run(self):
        for i, (video_idx, video) in enumerate(self.video_metadatas.iterrows()):
            log.info(
                f"Starting tracking on video ({i+1}/{len(self.video_metadatas)}) : {video.name}"
            )
            self.video_step(video, video_idx)

    def video_step(self, video, video_id):
        """Run tracking on a single video.

        Updates the tracking state, and creates the visualization for the
        current video.

        Args:
            video: a VideoMetadata series
            video_id: the video id. must be the same as video.name

        """
        raise NotImplementedError("OnlineTrackingEngine is broken now, please use OfflineTrackingEngine")  # FIXME refactor all tracking engine without pytorch lightning
        start = timer()
        imgs_meta = self.img_metadatas
        self.detection_datapipe.update(imgs_meta[imgs_meta.video_id == video_id])
        self.model_track.reset()

        start_process = timer()
        detections_list = self.trainer_model.predict(
            self, dataloaders=self.detection_dl
        )
        process_time = timer() - start_process
        detections = pd.concat(detections_list)
        self.tracker_state.update(detections)
        start_vis = timer()
        self.vis_engine.run(self.tracker_state, video_id)
        vis_time = timer() - start_vis
        self.tracker_state.free(video_id)
        video_time = timer() - start
        log.info(
            f"Completed video in {video_time:0.2f}s. (Detect+reid+track {process_time:0.2f}s, "
            + f"visualization {vis_time:0.2f}s and rest is {(video_time - process_time - vis_time):0.2f}s)"
        )

    def predict_step(self, batch, batch_idx):
        """Steps through tracking predictions for one image.

        This doesn't work for a batch or any other construct. To work on a batch
        we should modify the test_step to take a batch of images.

        Args:
            batch : a tuple containing the image indexes and the input as needed
                by model_detect
            batch_idx : the batch index (currently unused)
        Returns:
            detection_list : list of detection objects, with pose, reid and tracking info
        """
        idxs, batch = batch
        image_metadatas = self.img_metadatas.loc[idxs]

        # 1. Detection
        detections = Detections(self.model_detect.process(batch, image_metadatas))

        # 2. Reid
        reid_detections = []
        self.reid_datapipe.update(self.img_metadatas, detections)
        for idxs, reid_batch in self.reid_dl:
            batch_detections = detections.loc[idxs]
            batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
            reid_detections.append(
                self.model_reid.process(reid_batch, batch_detections, batch_metadatas)
            )

        if len(reid_detections) > 0:
            reid_detections = pd.concat(reid_detections)
        else:
            reid_detections = Detections()

        # 3. Tracking
        track_detections = []
        for image_id, image_metadata in image_metadatas.iterrows():
            image = cv2_load_image(image_metadata.file_path)
            self.model_track.prepare_next_frame(image)
            detections = reid_detections[reid_detections.image_id == image_id]
            self.track_datapipe.update(self.img_metadatas, detections)
            for idxs, track_batch in self.track_dl:
                batch_detections = detections.loc[idxs]
                batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
                track_detections.append(
                    self.model_track.process(
                        track_batch, image, batch_detections, batch_metadatas
                    )
                )

        if len(track_detections) > 0:
            track_detections = pd.concat(track_detections)
        else:
            track_detections = Detections()

        return track_detections


class OfflineTrackingEngine(OnlineTrackingEngine):
    """Offline tracking engine"""

    def video_step(self, video, video_id):
        start = timer()
        log.info(f"Starting tracking on video: {video_id}")
        with self.tracker_state(video_id) as tracker_state:
            self.model_track.reset()
            imgs_meta = self.img_metadatas[self.img_metadatas.video_id == video_id]
            detections = tracker_state.load()
            loaded = detections is not None
            detect_time = 0
            reid_time = 0
            tracking_time = 0

            if tracker_state.do_detection or not loaded:
                self.detection_datapipe.update(imgs_meta)
                model_detect = OfflineDetector(self.model_detect, imgs_meta)
                start_detect = timer()
                detections_list = self.trainer_model.predict(
                    model_detect, dataloaders=self.detection_dl
                )
                detect_time = timer() - start_detect
                detections = pd.concat(detections_list)
            if detections.empty:
                # video has not detections: nothing left to do ...
                return detections

            if tracker_state.do_reid or not loaded:
                self.reid_datapipe.update(imgs_meta, detections)
                model_reid = OfflineReider(self.model_reid, imgs_meta, detections)
                start_reid = timer()
                detections_list = self.trainer_model.predict(
                    model_reid, dataloaders=self.reid_dl
                )
                reid_time = timer() - start_reid
                detections = pd.concat(detections_list)

            if tracker_state.do_tracking or not loaded:
                track_detections = []
                for image_id in tqdm(imgs_meta.index):
                    start_track = timer()
                    image = cv2_load_image(imgs_meta.loc[image_id].file_path)
                    self.model_track.prepare_next_frame(image)
                    image_detections = detections[detections.image_id == image_id]
                    if len(detections) != 0:
                        self.track_datapipe.update(imgs_meta, image_detections)
                        model_track = OfflineTracker(
                            self.model_track, imgs_meta.loc[[image_id]], image_detections, image
                        )
                        detections_list = self.trainer_track.predict(
                            model_track, dataloaders=self.track_dl
                        )
                        track_detections += detections_list if detections_list else []
                    tracking_time += timer() - start_track

                if len(track_detections) > 0:
                    detections = pd.concat(track_detections)
                else:
                    detections = Detections()

            tracker_state.update(detections)
            tracker_state.save()
            start_vis = timer()
            self.vis_engine.run(tracker_state, video_id)
            vis_time = timer() - start_vis
            video_time = timer() - start
            log.info(
                f"Completed video in {video_time:0.2f}s. (Detect {detect_time:0.2f}s, reid "
                + f"{reid_time:0.2f}s, track {tracking_time:0.2f}s, visualization {vis_time:0.2f}s "
                + f"and rest {(video_time - detect_time - reid_time - tracking_time - vis_time):0.2f}s)"
            )


class OfflineDetector(pl.LightningModule):
    def __init__(self, model_detect, img_metadatas):
        super().__init__()
        self.model_detect = model_detect
        self.img_metadatas = img_metadatas

    def predict_step(self, batch, batch_idx: int):
        idxs, batch = batch
        image_metadatas = self.img_metadatas.loc[idxs]
        detections = Detections(self.model_detect.process(batch, image_metadatas))
        return detections


class OfflineReider(pl.LightningModule):
    def __init__(self, model_reid, img_metadatas, detections):
        super().__init__()
        self.model_reid = model_reid
        self.img_metadatas = img_metadatas
        self.detections = detections

    def predict_step(self, batch, batch_idx: int):
        idxs, batch = batch
        batch_detections = self.detections.loc[idxs]
        batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
        return self.model_reid.process(batch, batch_detections, batch_metadatas)


class OfflineTracker(pl.LightningModule):
    def __init__(self, model_track, img_metadatas, detections, image):
        super().__init__()
        self.model_track = model_track
        self.img_metadatas = img_metadatas
        self.detections = detections
        self.image = image

    def predict_step(self, batch, batch_idx: int):
        idxs, batch = batch
        batch_detections = self.detections.loc[idxs]
        batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
        return self.model_track.process(
            batch, self.image, batch_detections, batch_metadatas
        )
