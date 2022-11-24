from abc import ABC
from pathlib import Path

from pbtrack.tracker.categories import Categories
from pbtrack.tracker.detections import Detections
from pbtrack.tracker.images import Images


def assert_valid_columns(df, required_columns):
    assert set(df.columns).issuperset(required_columns), \
        f"Column {required_columns-set(df.columns)} is required to build a {df.__class__.__name__} DataFrame object"


class TrackingSet:
    def __init__(self, split: str, images: Images, categories: Categories, detections: Detections):
        assert_valid_columns(images, Images.required_columns)
        assert_valid_columns(categories, Categories.required_columns)
        assert_valid_columns(detections, Detections.required_columns)
        self.split = split
        self.images = images
        self.categories = categories
        self.detections = detections


class TrackingDataset(ABC):
    def __init__(self, name: str, nickname: str, dataset_path: str, train_set: TrackingSet, val_set: TrackingSet, test_set: TrackingSet, **kwargs):
        self.name = name
        self.nickname = nickname
        self.dataset_path = Path(dataset_path)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
