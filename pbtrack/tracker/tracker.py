from pbtrack.datasets.tracking_dataset import TrackingSet


class Tracker:
    def __init__(self, tracking_set: TrackingSet):
        self.gt = tracking_set
        self.predictions = None
