from abc import ABC
from pathlib import Path

from .metadatas import Metadatas
from .detections import Detections
from .categories import Categories

class TrackingSet:
    def __init__(self, split: str, metadatas: Metadatas, detections: Detections, categories: Categories):
        self.split = split
        self.metadatas = metadatas
        self.detections = detections
        self.categories = categories

class TrackingDataset(ABC):
    def __init__(self,
                 dataset_path: str,
                 train_set: TrackingSet,
                 val_set: TrackingSet,
                 test_set: TrackingSet,
                 *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
