from abc import abstractmethod, ABC
from pbtrack.datastruct.detections import Detections


# TODO Baptiste
class ReIdentifier(ABC):
    @abstractmethod
    def train(self, detections: Detections):
        pass

    @abstractmethod
    def run(self, detections: Detections):
        pass
