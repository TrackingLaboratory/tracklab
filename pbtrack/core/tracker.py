from typing import List
from abc import abstractmethod, ABC

from pbtrack.core.datastruct import Detection, Detections
from pbtrack.core.datastruct.image_metadatas import ImageMetadata


class Tracker(ABC):
    """Abstract class to implement for the integration of a new tracking algorithm
    in wrappers/track. The functions to implement are __init__, reset
    (optional), preprocess and process. A description of the expected
    behavior is provided below.
    """

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
