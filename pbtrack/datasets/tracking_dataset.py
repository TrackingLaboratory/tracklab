from pbtrack.tracker.categories import Categories
from pbtrack.tracker.detections import Detections
from pbtrack.tracker.images import Images


class TrackingSet:
    def __init__(self, images: Images, categories: Categories, detections: Detections):
        self.images = images
        self.categories = categories
        self.dets_gt = detections


class TrackingDataset:
    def __init__(self, dataset_path: str, train_set: TrackingSet, val_set: TrackingSet, test_set: TrackingSet, **kwargs):
        self.dataset_path = dataset_path
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
