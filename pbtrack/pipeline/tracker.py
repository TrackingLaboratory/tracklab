from typing import List, Any, Union
from abc import abstractmethod, ABC

import pandas as pd
from torch.utils.data import DataLoader

import pbtrack
from pbtrack.engine import TrackingEngine, EngineDatapipe
from pbtrack.utils.cv2 import cv2_load_image
from . import Module


class Tracker(Module):
    """Abstract class to implement for the integration of a new detector in wrappers/track.
    The functions to implement are __init__, preprocess and process.
    A description of the expected behavior is provided below.
    """

    def __init__(self, cfg, device, batch_size):
        """Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the detector
            device (int): device to use for the detector
            batch_size (int): batch size for the detector
        """
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self._datapipe = None

    def process_video(
        self, detections: pd.DataFrame, imgs_meta: pd.DataFrame, engine: TrackingEngine
    ) -> pd.DataFrame:
        for image_id in imgs_meta.index:
            image = cv2_load_image(imgs_meta.loc[image_id].file_path)
            self.prepare_next_frame(image)
            image_detections = detections[detections.image_id == image_id]

            if len(image_detections) != 0:
                self.datapipe.update(imgs_meta, image_detections)
                for batch in self.dataloader():
                    detections = engine.default_step(batch, self.name, detections, image=image)
        return detections

    @abstractmethod
    def preprocess(self, detection: pd.Series, metadata: pd.Series) -> object:
        """Your preprocessing function to adapt the input to your tracker.
        The output is being batched by the collate_fn() and fed to process().
        Args:
            detection (pd.Series): the detection to process
            metadata (pd.Series): the image metadata to process
        Returns:
            preprocessed (Any): preprocessed input for the process step
        """
        pass

    @abstractmethod
    def process(
        self, batch: Any, detections: pd.DataFrame
    ) -> Union[pd.Series, List[pd.Series], pd.DataFrame, List[pd.DataFrame]]:
        """Your processing function to run the detector
        Args:
            batch (Any): the batched outputs from preprocess() by collate_fn()
            detections (pd.DataFrame): the images metadata associated to the batch
        Returns:
            detections (Union[pd.Series, List[pd.Series], pd.DataFrame, List[pd.DataFrame]]): the new detections
            from the batch. The framework will aggregate automatically all the results according to the `name` of the
            Series/`index` of the DataFrame. It is thus mandatory here to name correctly your series or index your
            dataframes. The output will override the previous detections with the same name/index.
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
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, **kwargs):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=2**16,
            num_workers=0,
            persistent_workers=False,
        )
