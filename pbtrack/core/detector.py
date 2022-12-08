from typing import List
from abc import abstractmethod, ABC

from pbtrack.datastruct.detections import Detection
from pbtrack.datastruct.images import Image, Images

class Detector(ABC):
    """ Abstract class to implement for the integration of a new detector
        in wrapper/detect. The functions to implement are __init__, train 
        (optional), pre_process and process. A description of the expected 
        behavior is provided below.
    """
    @abstractmethod
    def __init__(self, cfg, device):
        """ Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the detector
            device (str): device to use for the detector        
        """
        self.cfg = cfg
        self.device = device
    
    @abstractmethod
    def pre_process(self, image: Image) -> object:
        """ Your pre-processing function to adapt the input to your detector
        Args:
            image (Image): the image metadata to process
        Returns:
            pre_processed (object): pre_processed input for process()
        """
        pass
    
    @abstractmethod
    def process(self, pre_processed_batch, images: Images) -> List[Detection]:
        """ Your processing function to run the detector
        Args:
            pre_processed_batch (object): output of pre_process() by batch
            images (Images): the images metadata associated to the batch
        Returns:
            detections (List[Detection]): list of new detections for the batch
        """
        pass

    def train(self):
        """ Training function for your detector
        """
        pass