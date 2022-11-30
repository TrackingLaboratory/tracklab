import torch
from typing import List
from abc import abstractmethod, ABC
from pbtrack.datastruct.detections import Detection
from pbtrack.datastruct.images import ImagesSeries

# TODO check
class Detector(ABC):
    """
        abstract class to implement for the integration of a new detector 
        in wrapper/detect. The functions to implement are __init__, train, 
        pre_process, __call__ and post_process. A description of the expected 
        behavior is provided below.
    """
    @abstractmethod
    def __init__(self, cfg, device):
        """
        Args:
            cfg (NameSpace): configuration file from Hydra for the detector
            device (str): device to use for the detector            
        """
        self.cfg = cfg
        self.device = device
    
    @abstractmethod
    def train(self):
        """training function for the detector
        """
        pass
    
    def pre_process(self, image: torch.Tensor) -> object:
        """ Your pre_processing function to adapt the input to your detector
        Args:
            image (torch.Tensor): the image to process
        Returns:
            pre_processed (object): pre_processed input for process function
        """
        return image
    
    @abstractmethod
    def process(self, results) -> object:
        """ Your processing function to run the detector
        Args:
            processed (object): output of pre_process()
        Returns:
            results (object): your detection results
        """
        pass
    
    @abstractmethod
    def post_process(self, results, Image: ImagesSeries) -> List[Detection]:
        """ Your post processing function to adapt the the output of process
        to a list of Detection objects
        Args:
            results (object): output of process function
            Image (ImagesSeries): metadata of the corresponding image
        Returns:
            detections (List[Detection]): new detections
        """
        pass
    
    def __call__(self, image: torch.Tensor, Image: ImagesSeries) -> List[Detection]:
        """process the detector pipeline
        Args:
            image (torch.Tensor): the image to process
            Image (ImagesSeries): metadata of the corresponding image
        Returns:
            detections (List[Detection]): new detections
        """
        pre_processed = self.pre_process(image)
        results = self.process(pre_processed)
        return self.post_process(results, Image)
