from pbtrack.datastruct import Detections
from pbtrack.engine import TrackingEngine


class PipelinedTrackingEngine(TrackingEngine):
    """ Pipelined implementation of an online tracking engine."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def video_loop(self, video, video_id) -> "Detections":
        pass