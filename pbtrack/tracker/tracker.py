from pbtrack.datasets.tracking_dataset import TrackingSet


class Tracker:
    # TODO add functions train/val/test or split(train/val/test)
    def __init__(self, tracking_set: TrackingSet):
        self.gt = tracking_set
        self.predictions = None
