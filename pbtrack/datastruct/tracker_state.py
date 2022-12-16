import pandas as pd

from .tracking_dataset import TrackingSet


class TrackerState:
    def __init__(self, tracking_set: TrackingSet):
        self.gt = tracking_set
        self.predictions = None

    def update(self, detections):
        if self.predictions is None:
            self.predictions = detections
        else:
            self.predictions = pd.concat([self.predictions, detections])
