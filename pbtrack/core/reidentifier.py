from typing import List
from abc import abstractmethod, ABC

from torch.utils.data import DataLoader

import pbtrack
from pbtrack.datastruct.image_metadatas import ImageMetadata
from pbtrack.datastruct.detections import Detection, Detections
from ..utils.collate import default_collate


class ReIdentifier(ABC):
    """Abstract class to implement for the integration of a new reidentifier
    in wrappers/reid. The functions to implement are __init__, train
    (optional), preprocess and process. A description of the expected
    behavior is provided below.
    """

    @abstractmethod
    def __init__(self, cfg, device, batch_size):
        """Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the reidentifier
            device (str): device to use for the reidentifier
        """
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, detection: Detection, metadata: ImageMetadata) -> object:
        """Your pre-processing function to adapt the input to your reidentifier
        Args:
            detection (Detection): the detection to process
            metadata (ImageMetadata): the image metadata associated to the detection
        Returns:
            preprocessed (object): preprocessed input for process()
        """
        pass

    @abstractmethod
    def process(self, batch, detections: Detections) -> List[Detection]:
        """Your processing function to run the reidentifier
        Args:
            batch (object): output of preprocess() by batch
            detections (Detections): the detections to update
        Returns:
            detections (List[Detection]): updated detections for the batch
        """
        pass

    def train(self):
        """Training function for your reidentifier"""
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = pbtrack.EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "pbtrack.TrackingEngine"):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            num_workers=engine.num_workers,
            persistent_workers=False,
        )
