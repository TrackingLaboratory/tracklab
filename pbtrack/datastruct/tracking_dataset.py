from abc import ABC
from pathlib import Path

from .detections import Detections
from .image_metadatas import ImageMetadatas
from .video_metadatas import VideoMetadatas

class TrackingSet:
    def __init__(self, split: str, detections: Detections, image_metadatas: ImageMetadatas, video_metadatas: VideoMetadatas):
        self.split = split
        self.detections = detections
        self.image_metadatas = image_metadatas
        self.video_metadatas = video_metadatas

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
