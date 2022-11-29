from abc import abstractmethod, ABC
from pbtrack.datastruct.detections import Detections, DetectionsSeries

# TODO check
class Detector(ABC):
    """
        abstract class to implement for the integration of a new detector 
        in wrapper/detect. The functions to implement are __init__, train, 
        pre_process and post_process. A description of the expected behavior 
        is provided below.
    """
    @abstractmethod
    def __init__(self, cfg, device):
        """
        Args:
            cfg (NameSpace): configuration file for the detector
            device (str): device to use for the detector
            
        Attributes:
            model (nn.Module): model of the detector
        """
        self.cfg = cfg
        self.device = device
        self.model = ...
    
    @abstractmethod
    def train(self):
        """TODO implement this function to train the detector
        """
        pass
    
    @abstractmethod
    def pre_process(self, Detection: DetectionsSeries) -> object:
        """ Your pre_processing function to adapt the input to your detector
        Args:
            Detection (DetectionsSeries): a single detection object
        Returns:
            pre_processed (object): pre_processed input for self.model
        """
        pass
    
    def __call__(self, pre_processed):
        """process the pre_processed data and return the data
        Args:
            pre_processed (object): output of pre_process function
        Returns:
            processed (object): output of self.model
        """
        processed = self.model(pre_processed)  # type: ignore
        return processed
    
    @abstractmethod
    def post_process(self, processed) -> Detections:
        """ Your post processing function to adapt the output of self.model
        to a Detections object
        Args:
            processed (object): output of self.model(pre_processed)
        Returns:
            Detections (pd.DataFrame): new detections
        """
        pass
