from typing import List
from abc import abstractmethod, ABC

from pbtrack.datastruct.detections import Detection, Detections
from pbtrack.datastruct.images import Image

class ReIdentifier(ABC):
    """ Abstract class to implement for the integration of a new reidentifier
        in wrapper/reid. The functions to implement are __init__, train 
        (optional), pre_process and process. A description of the expected 
        behavior is provided below.
    """
    @abstractmethod
    def __init__(self, cfg, device):
        """ Init function
        Args:
            cfg (NameSpace): configuration file from Hydra for the reidentifier
            device (str): device to use for the reidentifier        
        """
        self.cfg = cfg
        self.device = device
    
    @abstractmethod
    def pre_process(self, detection: Detection, image: Image)-> object:
        """ Your pre-processing function to adapt the input to your 
            reidentifier
        Args:
            detection (Detection): the detection to process
            image (Image): the image metadata associated to the detection
        Returns:
            pre_processed (object): pre_processed input for process()
        """
        pass
        # return pre_processed tel que collate_fn is OK nativement
    
    @abstractmethod
    def process(self, pre_processed_batch, detections: Detections) -> List[Detection]:
        """ Your processing function to run the reidentifier
        Args:
            pre_processed_batch (object): output of pre_process() by batch
            detections (Detections): the detections to update
        Returns:
            detections (List[Detection]): updated detections for the batch
        """
        pass
    
    def train(self):
        """ Training function for your reidentifier
        """
        pass