from typing import List
from abc import abstractmethod, ABC

import pandas as pd
from torch.utils.data import DataLoader

import pbtrack
from pbtrack.datastruct.image_metadatas import ImageMetadata
from pbtrack.datastruct.detections import Detection, Detections
from pbtrack.engine import TrackingEngine
from pbtrack.utils.images import cv2_load_image


class Tracker(ABC):
    """Abstract class to implement for the integration of a new tracking algorithm
    in wrappers/track. The functions to implement are __init__, reset
    (optional), preprocess and process. A description of the expected
    behavior is provided below.
    """

    def __init__(self, cfg, device, batch_size):
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self._datapipe = None

    def process_video(self, detections, imgs_meta, engine: TrackingEngine) -> Detections:
        track_detections = []
        for image_id in imgs_meta.index:
            image = cv2_load_image(imgs_meta.loc[image_id].file_path)
            self.prepare_next_frame(image)
            image_detections = detections[detections.image_id == image_id]

            if len(image_detections) != 0:
                self.datapipe.update(imgs_meta, image_detections)
                for batch in self.dataloader():
                    track_detections.append(engine.track_step(batch, detections, image))
        if len(track_detections) > 0:
            return pd.concat(track_detections)
        else:
            return Detections()

    @abstractmethod
    def preprocess(self, detection: Detection, metadata: ImageMetadata) -> object:
        """Your preprocessing function to adapt the input to your tracking algorithm
        Args:
            detection (Detection): the detection to process
            metadata (ImageMetadata): the image metadata associated to the detection
        Returns:
            preprocessed (object): preprocessed input for process()
        """
        pass

    @abstractmethod
    def process(self, preprocessed_batch, detections: Detections) -> List[Detection]:
        """Your processing function to run the tracking algorithm
        Args:
            preprocessed_batch (object): output of preprocess() by batch
            detections (Detections): the detections to update
        Returns:
            detections (List[Detection]): updated detections for the batch
        """
        pass

    def reset(self):
        """Reset the tracker state to start tracking in a new video"""
        pass

    def prepare_next_frame(self, next_frame):
        """Prepare tracker for the next frame, can be used to propagate state distribution with KF, doing camera motion
        compensation, etc."""
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = pbtrack.EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, **kwargs):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=2**16,
            num_workers=0,
            persistent_workers=False,
        )


# FIXME a bit useless no ?
class OfflineTracker(Tracker):
    @abstractmethod
    def run(self, video_dets: Detections):
        # update video_dets
        pass


# FIXME a bit useless no ?
class OnlineTracker(Tracker):
    def reset(self):
        pass
