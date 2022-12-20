from typing import List
from torch.utils.data.dataloader import default_collate
from abc import abstractmethod, ABC
from pbtrack.core.datastruct import Detection
from pbtrack.core.datastruct.image_metadatas import ImageMetadata, ImageMetadatas


class Detector(ABC):
    """Abstract class to implement for the integration of a new detector
    in wrapper/detect. The functions to implement are __init__, train
    (optional), preprocess and process. A description of the expected
    behavior is provided below.
    """

    collate_fn = default_collate

    @abstractmethod
    def __init__(self, cfg, device):
        """Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the detector
            device (str): device to use for the detector
        Attributes:
            id (int): id of the detection
        """
        self.cfg = cfg
        self.device = device
        self.id = 0

    @abstractmethod
    def preprocess(self, metadata: ImageMetadata) -> object:
        """Your preprocessing function to adapt the input to your detector
        Args:
            image (ImageMetadata): the image metadata to process
        Returns:
            preprocessed (object): preprocessed input for process()
        """
        pass

    @abstractmethod
    def process(self, preprocessed_batch, metadatas: ImageMetadatas) -> List[Detection]:
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
