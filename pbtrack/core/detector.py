from typing import List, Union, Tuple
from abc import abstractmethod, ABC

from torch.utils.data.dataloader import default_collate

from .datastruct.detections import Detection
from .datastruct.image_metadatas import ImageMetadata, ImageMetadatas


class Detector(ABC):
    """Abstract class to implement for the integration of a new detector
    in wrappers/detect. The functions to implement are __init__, train
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
            id (int): id of the current detection
        """
        self.cfg = cfg
        self.device = device
        self.id = 0

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
    ) -> Union[List[Detection], Tuple[List[Detection], object]]:
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
