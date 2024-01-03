import pandas as pd

from tracklab.engine import TrackingEngine


class PipelinedTrackingEngine(TrackingEngine):
    """ Pipelined implementation of an online tracking engine."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for name, model in self.models.items():
            pass

    def video_loop(self, video, video_id) -> pd.DataFrame:
        pass