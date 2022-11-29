from abc import abstractmethod, ABC
from pbtrack.datastruct.detections import Detections
from pbtrack.datastruct.tracker_state import TrackerState


# TODO Baptiste
class Detector(ABC):
    @abstractmethod
    def train(self, detections: Detections):
        pass

    @abstractmethod
    def run(self, tracker_state: TrackerState):
        pass
