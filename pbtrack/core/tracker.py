from abc import abstractmethod, ABC
from pbtrack.datastruct import Detections, Detection, ImageMetadata, ImageMetadatas


class Tracker(ABC):
    @abstractmethod
    def preprocess(self, detection: Detection, metadata: ImageMetadata):
        pass

    @abstractmethod
    def process(self, batch, detections: Detections, metadatas: ImageMetadatas):
        pass

    @abstractmethod
    def reset(self):
        """ Reset the tracker state to start tracking in a new video."""
        pass


class OfflineTracker(Tracker):
    @abstractmethod
    def run(self, video_dets: Detections):
        # update video_dets
        pass


class OnlineTracker(Tracker):
    def reset(self):
        pass
