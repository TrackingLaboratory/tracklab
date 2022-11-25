from abc import ABC
from pathlib import Path
from pbtrack.tracker.categories import Categories
from pbtrack.tracker.detections import Detections
from pbtrack.tracker.images import Images


def assert_valid_columns(df, rcols):
    required_columns = {k for k, v in rcols.items() if v}
    assert set(df.columns).issuperset(required_columns), \
        f"Column {required_columns-set(df.columns)} is required to build a {df.__class__.__name__} DataFrame object"


class TrackingSet:
    def __init__(self, split: str, images: Images, categories: Categories, detections: Detections):
        assert_valid_columns(images, Images.rcols)
        assert_valid_columns(categories, Categories.rcols)
        assert_valid_columns(detections, Detections.rcols)
        self.images = images.reindex(columns=Images.rcols.keys())
        self.categories = categories.reindex(columns=Categories.rcols.keys())
        self.detections = detections.reindex(columns=Detections.rcols.keys())
        self.split = split


class TrackingDataset(ABC):
    def __init__(self, name: str, nickname: str, dataset_path: str, train_set: TrackingSet, val_set: TrackingSet, test_set: TrackingSet, **kwargs):
        self.name = name
        self.nickname = nickname
        self.dataset_path = Path(dataset_path)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
