from abc import ABC
from pathlib import Path
from dataclasses import dataclass

import pandas as pd


@dataclass
class TrackingSet:
    video_metadatas: pd.DataFrame
    image_metadatas: pd.DataFrame
    detections_gt: pd.DataFrame


class TrackingDataset(ABC):
    def __init__(
        self,
        dataset_path: str,
        train_set: TrackingSet,
        val_set: TrackingSet,
        test_set: TrackingSet,
        nvid: int = -1,
        nframes: int = -1,
        vids_dict: list = None,
        *args,
        **kwargs
    ):
        self.dataset_path = Path(dataset_path)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        if self.train_set is not None:
            self.train_set = self._subsample(
                self.train_set, nvid, nframes, getattr(vids_dict, "train", None)
            )
        if self.val_set is not None:
            self.val_set = self._subsample(
                self.val_set, nvid, nframes, getattr(vids_dict, "val", None)
            )
        if self.test_set is not None:
            self.test_set = self._subsample(
                self.test_set, nvid, nframes, getattr(vids_dict, "test", None)
            )

    def _subsample(self, tracking_set, nvid, nframes, vids_names):
        if nvid < 1 and nframes < 1 and (vids_names is None or len(vids_names) == 0):
            return tracking_set

        # filter videos:
        if vids_names is not None and len(vids_names) > 0:  # keep videos in vids_dict
            videos_to_keep = tracking_set.video_metadatas[
                tracking_set.video_metadatas.name.isin(vids_names)
            ].index
            tiny_video_metadatas = tracking_set.video_metadatas.loc[videos_to_keep]
        elif nvid > 0:  # keep 'nvid' videos
            videos_to_keep = tracking_set.video_metadatas.sample(
                nvid, random_state=2
            ).index
            tiny_video_metadatas = tracking_set.video_metadatas.loc[videos_to_keep]
        else:  # keep all videos
            videos_to_keep = tracking_set.video_metadatas.index
            tiny_video_metadatas = tracking_set.video_metadatas

        # filter images:
        # keep only images from videos to keep
        tiny_image_metadatas = tracking_set.image_metadatas[
            tracking_set.image_metadatas.video_id.isin(videos_to_keep)
        ]

        # keep only images from first nframes
        if nframes > 0:
            tiny_image_metadatas = tiny_image_metadatas.groupby("video_id").head(
                nframes
            )

        # filter detections:
        tiny_detections = None
        if tracking_set.detections_gt is not None:
            tiny_detections = tracking_set.detections_gt[
                tracking_set.detections_gt.image_id.isin(tiny_image_metadatas.index)
            ]

        return TrackingSet(
            tiny_video_metadatas,
            tiny_image_metadatas,
            tiny_detections,
        )
