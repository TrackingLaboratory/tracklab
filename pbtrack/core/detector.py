from typing import List, Union, Tuple, Any
from abc import abstractmethod, ABC

from ..datastruct import Detection, Detections
from ..engine import TrackingEngine

from torch.utils.data.dataloader import default_collate, DataLoader

from pbtrack.datastruct.image_metadatas import ImageMetadata, ImageMetadatas
from .datapipe import EngineDatapipe


class Detector(ABC):
    """Abstract class to implement for the integration of a new detector
    in wrappers/detect. The functions to implement are __init__, train
    (optional), preprocess and process. A description of the expected
    behavior is provided below.
    """

    collate_fn = default_collate

    @abstractmethod
    def __init__(self, cfg, device, batch_size):
        """Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the detector
            device (int): device to use for the detector
        Attributes:
            id (int): id of the current detection
        """
        self.cfg = cfg
        self.device = device
        self.id = 0
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, metadata: ImageMetadata) -> object:
        """Your preprocessing function to adapt the input to your detector
        Args:
            metadata (ImageMetadata): the image metadata to process
        Returns:
            preprocessed (object): preprocessed input for process()
        """
        pass

    @abstractmethod
    def process(
        self, preprocessed_batch, metadatas: ImageMetadatas, return_fields=False
    ) -> Union[List["Detection"], Tuple[List["Detection"], object]]:
        """Your processing function to run the detector
        Args:
            preprocessed_batch (object): output of preprocess() by batch
            metadatas (ImageMetadatas): the images metadata associated to the batch
        Returns:
            detections (List[Detection]): list of new detections for the batch
        """
        pass

    def train(self):
        """Training function for your detector"""
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "TrackingEngine"):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=self.batch_size,
            collate_fn=type(self).collate_fn,  # not using type(self) fails !
            num_workers=engine.num_workers,
            persistent_workers=False,
        )


class SingleDetector(ABC):
    """Detector that takes a bounding box and returns a bounding box per bbox.

    The API is exactly the same as the ReIdentifier, but this makes it clear
    that there is a difference between both.
    """
    collate_fn = default_collate

    @abstractmethod
    def __init__(self, cfg, device, batch_size):
        """Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the detector
            device (str): device to use for the detector
            batch_size (int) : batch size for the dataloader
        """
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, detection: "Detection",
                   metadata: ImageMetadata) -> Any:
        """Your pre-processing function to adapt the input to your detector
        Args:
            detection (Detection): the detection to process
            metadata (ImageMetadata): the image metadata associated to the detection
        Returns:
            preprocessed (object): preprocessed input for process()
        """
        pass

    @abstractmethod
    def process(self, batch: Any,
                detections: "Detections") -> List["Detection"]:
        """Your processing function to run the detector
        Args:
            batch (object): output of preprocess() by batch
            detections (Detections): the detections to update
        Returns:
            detections (List[Detection]): updated detections for the batch
        """
        pass

    def train(self):
        """Training function for your detector"""
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "TrackingEngine"):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=self.batch_size,
            collate_fn=type(self).collate_fn,
            num_workers=engine.num_workers,
            persistent_workers=False,
        )