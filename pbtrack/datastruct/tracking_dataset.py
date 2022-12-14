from abc import ABC
from pathlib import Path
from .detections import Detections
from .image_metadatas import ImageMetadatas
from .video_metadatas import VideoMetadatas


class TrackingSet:
    def __init__(
        self,
        split: str,
        video_metadatas: VideoMetadatas,
        image_metadatas: ImageMetadatas,
        detections: Detections,
    ):
        self.split = split
        self.video_metadatas = video_metadatas
        self.image_metadatas = image_metadatas
        self.detections = detections


class TrackingDataset(ABC):
    def __init__(
        self,
        dataset_path: str,
        train_set: TrackingSet,
        val_set: TrackingSet,
        test_set: TrackingSet,
        nvid: int = -1,
        nframes: int = -1,
        *args,
        **kwargs
    ):
        self.dataset_path = Path(dataset_path)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        if nvid > 0 or nframes > 0:
            self.train_set = self._subsample(self.train_set, nvid, nframes)
            self.val_set = self._subsample(self.val_set, nvid, nframes)

    def _subsample(self, tracking_set, nvid=2, nframes=5):
        # filter videos:
        videos_to_keep = tracking_set.video_metadatas.sample(nvid, random_state=3).index
        tiny_video_metadatas = tracking_set.video_metadatas.loc[videos_to_keep]

        # filter images:
        # keep only images from videos to keep
        tiny_image_metadatas = tracking_set.image_metadatas[
            tracking_set.image_metadatas.video_id.isin(videos_to_keep)
        ]

        # keep only images from first nframes
        tiny_image_metadatas = tiny_image_metadatas.groupby("video_id").head(nframes)

        # filter detections:
        tiny_detections = tracking_set.detections[
            tracking_set.detections.image_id.isin(tiny_image_metadatas.index)
        ]

        return TrackingSet(
            tracking_set.split,
            tiny_video_metadatas,
            tiny_image_metadatas,
            tiny_detections,
        )
