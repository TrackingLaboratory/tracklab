import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from timeit import default_timer as timer

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pbtrack.core import Detector, ReIdentifier, Tracker, EngineDatapipe
from pbtrack.core.datastruct.tracker_state import TrackerState
from pbtrack.core.datastruct import Detections
from pbtrack.visualization.visualization_engine import VisualisationEngine
from pbtrack.utils.collate import default_collate
from pbtrack.utils.images import cv2_load_image

import logging

log = logging.getLogger(__name__)

import warnings

warnings.filterwarnings(
    "ignore", ".*does not have many workers.*"
)  # Disable UserWarning for DataLoaders with num_workers=0


class OnlineTrackingEngine(pl.LightningModule):
    """Online tracking engine

    Args:
        detect_multi_model: Predicts pose detection
        detect_single_model: Predicts single pose detection
        reid_model: Predicts reid embeddings
        track_model: Tracking algorithm
        tracker_state: Running state of the tracking
        vis_engine: Visualisation engine
        detect_multi_batchsize: Batch size for multi detection
        detect_single_batchsize: Batch size for single detection
        reid_batchsize: Batch size for reid
        num_workers: Number of workers for dataloaders
    """

    def __init__(
        self,
        detect_multi_model: Detector,
        detect_single_model: Detector,
        reid_model: ReIdentifier,
        track_model: Tracker,
        tracker_state: TrackerState,
        vis_engine: VisualisationEngine,
        detect_multi_batchsize: int,
        detect_single_batchsize: int,
        reid_batchsize: int,
        num_workers: int,
    ):
        super().__init__()
        self.detect_is_top_down = type(detect_single_model) != DictConfig

        self.trainer_model = pl.Trainer()
        self.trainer_track = pl.Trainer(enable_progress_bar=False)
        self.vis_engine = vis_engine  # TODO : Convert to callback
        self.tracker_state = tracker_state
        self.img_metadatas = tracker_state.gt.image_metadatas
        self.video_metadatas = tracker_state.gt.video_metadatas

        self.detect_multi_model = detect_multi_model
        self.detect_single_model = detect_single_model
        self.reid_model = reid_model
        self.track_model = track_model

        self.detect_multi_batchsize = detect_multi_batchsize
        self.detect_single_batchsize = detect_single_batchsize
        self.reid_batchsize = reid_batchsize

        self.multi_detect_datapipe = EngineDatapipe(self.detect_multi_model)
        self.multi_detect_dl = DataLoader(
            dataset=self.multi_detect_datapipe,
            batch_size=self.detect_multi_batchsize,
            collate_fn=type(self.detect_multi_model).collate_fn,
            num_workers=num_workers,
            persistent_workers=False,
        )
        if self.detect_is_top_down:
            self.single_detect_datapipe = EngineDatapipe(self.detect_single_model)
            self.single_detect_dl = DataLoader(
                dataset=self.single_detect_datapipe,
                batch_size=self.detect_single_batchsize,
                collate_fn=type(self.detect_single_model).collate_fn,
                num_workers=num_workers,
                persistent_workers=False,
            )
        self.reid_datapipe = EngineDatapipe(self.reid_model)
        self.reid_dl = DataLoader(
            dataset=self.reid_datapipe,
            batch_size=self.reid_batchsize,
            num_workers=num_workers,
            collate_fn=default_collate,
            persistent_workers=False,
        )
        self.track_datapipe = EngineDatapipe(self.track_model)
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
        # FIXME TODO make it work again
        # FIXME refactor all tracking engine without pytorch lightning
        raise NotImplementedError(
            "OnlineTrackingEngine is broken now, please use OfflineTrackingEngine"
        )
        start = timer()
        imgs_meta = self.img_metadatas
        self.detection_datapipe.update(imgs_meta[imgs_meta.video_id == video_id])
        self.track_model.reset()

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
                self.reid_model.process(reid_batch, batch_detections, batch_metadatas)
            )

        if len(reid_detections) > 0:
            reid_detections = pd.concat(reid_detections)
        else:
            reid_detections = Detections()

        # 3. Tracking
        track_detections = []
        for image_id, image_metadata in image_metadatas.iterrows():
            image = cv2_load_image(image_metadata.file_path)
            self.track_model.prepare_next_frame(image)
            detections = reid_detections[reid_detections.image_id == image_id]
            self.track_datapipe.update(self.img_metadatas, detections)
            for idxs, track_batch in self.track_dl:
                batch_detections = detections.loc[idxs]
                batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
                track_detections.append(
                    self.track_model.process(
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
            self.track_model.reset()
            imgs_meta = self.img_metadatas[self.img_metadatas.video_id == video_id]
            detections = tracker_state.load()
            loaded = detections is not None
            detect_multi_time = 0
            detect_single_time = 0
            reid_time = 0
            tracking_time = 0

            if (
                tracker_state.do_detection or not loaded
            ):  # FIXME adapt here for single and multiple detection
                self.multi_detect_datapipe.update(imgs_meta)
                multi_model_detect = OfflineDetector(self.detect_multi_model, imgs_meta)
                start_multi_detect = timer()
                detections_list = self.trainer_model.predict(
                    multi_model_detect, dataloaders=self.multi_detect_dl
                )
                detect_multi_time = timer() - start_multi_detect
                detections = pd.concat(detections_list)
            if detections.empty:
                # FIXME not sure it is the right place to do this
                # The tracker might want to update the state even if there is no detection
                # video has not detections: nothing left to do ...
                return detections

            if self.detect_is_top_down and (
                tracker_state.do_detection or not loaded
            ):  # FIXME do the tracker_state for single_detection
                self.single_detect_datapipe.update(imgs_meta, detections)
                single_model_detect = OfflineReider(
                    self.detect_single_model, imgs_meta, detections
                )  # FIXME change name of pl module
                start_single_detect = timer()
                detections_list = self.trainer_model.predict(
                    single_model_detect, dataloaders=self.single_detect_dl
                )
                detect_single_time = timer() - start_single_detect
                detections = pd.concat(detections_list)

            if tracker_state.do_reid or not loaded:
                self.reid_datapipe.update(imgs_meta, detections)
                reid_model = OfflineReider(self.reid_model, imgs_meta, detections)
                start_reid = timer()
                detections_list = self.trainer_model.predict(
                    reid_model, dataloaders=self.reid_dl
                )
                reid_time = timer() - start_reid
                detections = pd.concat(detections_list)

            # Why isn't it done into a .predict() or .step() to keep
            # the same way of doing things as before ?
            if tracker_state.do_tracking or not loaded:
                track_detections = []
                for image_id in tqdm(imgs_meta.index):
                    start_track = timer()
                    image = cv2_load_image(imgs_meta.loc[image_id].file_path)
                    self.track_model.prepare_next_frame(image)
                    image_detections = detections[detections.image_id == image_id]
                    if len(image_detections) != 0:
                        self.track_datapipe.update(imgs_meta, image_detections)
                        track_model = OfflineTracker(
                            self.track_model,
                            imgs_meta.loc[[image_id]],
                            image_detections,
                            image,
                        )
                        detections_list = self.trainer_track.predict(
                            track_model, dataloaders=self.track_dl
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
            log.info(  # FIXME make it more readable
                f"Completed video in {video_time:0.2f}s. (Detect {detect_multi_time + detect_single_time:0.2f}s, reid "
                + f"{reid_time:0.2f}s, track {tracking_time:0.2f}s, visualization {vis_time:0.2f}s "
                + f"and rest {(video_time - detect_multi_time - detect_single_time - reid_time - tracking_time - vis_time):0.2f}s)"
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
    def __init__(self, reid_model, img_metadatas, detections):
        super().__init__()
        self.reid_model = reid_model
        self.img_metadatas = img_metadatas
        self.detections = detections

    def predict_step(self, batch, batch_idx: int):
        idxs, batch = batch
        batch_detections = self.detections.loc[idxs]
        batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
        return self.reid_model.process(batch, batch_detections, batch_metadatas)


class OfflineTracker(pl.LightningModule):
    def __init__(self, track_model, img_metadatas, detections, image):
        super().__init__()
        self.track_model = track_model
        self.img_metadatas = img_metadatas
        self.detections = detections
        self.image = image

    def predict_step(self, batch, batch_idx: int):
        idxs, batch = batch
        batch_detections = self.detections.loc[idxs]
        batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
        return self.track_model.process(
            batch, self.image, batch_detections, batch_metadatas
        )
