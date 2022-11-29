from abc import abstractmethod, ABC
from pbtrack.datastruct.detections import Detections


# TODO Baptiste
class Tracker(ABC):
    @abstractmethod
    def run(self, detections: Detections):
        pass


class OfflineTracker(Tracker):
    @abstractmethod
    def run(self, video_dets: Detections):
        # update video_dets
        pass


class OnlineTracker(Tracker):
    @abstractmethod
    def run(self, frame_dets: Detections):
        # update frame_dets
        pass

    def reset(self):
        pass
