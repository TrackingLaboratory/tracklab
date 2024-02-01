from abc import ABC
from pathlib import Path
from dataclasses import dataclass

import pandas as pd


@dataclass
class TrackingSet:
    video_metadatas: pd.DataFrame
    image_metadatas: pd.DataFrame
    detections_gt: pd.DataFrame
    image_gt: pd.DataFrame = pd.DataFrame(columns=["video_id"])


class TrackingDataset(ABC):
    def __init__(
        self,
        dataset_path: str,
        sets: dict[str, TrackingSet],
        nvid: int = -1,
        nframes: int = -1,
        vids_dict: list = None,
        *args,
        **kwargs
    ):
        self.dataset_path = Path(dataset_path)
        self.sets = sets
        self.train_set = None
        self.val_set = None
        self.test_set = None

        sub_sampled_sets = {}
        for set_name, split in self.sets.items():
            vid_list = vids_dict[set_name] if vids_dict is not None and set_name in vids_dict else None
            sub_sampled_sets[set_name] = self._subsample(split, nvid, nframes, vid_list)
        self.sets = sub_sampled_sets

    def _subsample(self, tracking_set, nvid, nframes, vids_names):
        if nvid < 1 and nframes < 1 and (vids_names is None or len(vids_names) == 0) or tracking_set is None:
            return tracking_set

        # filter videos:
        if vids_names is not None and len(vids_names) > 0:
            assert set(vids_names).issubset(tracking_set.video_metadatas.name.unique()), f"Some videos to process {set(vids_names) - set(tracking_set.video_metadatas.name.unique())} does not exist in the tracking set"
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
        tiny_image_gt = tracking_set.image_gt[
            tracking_set.image_gt.video_id.isin(videos_to_keep)
        ]

        # keep only images from first nframes
        if nframes > 0:
            tiny_image_metadatas = tiny_image_metadatas.groupby("video_id").head(
                nframes
            )
            tiny_image_gt = tiny_image_gt.groupby("video_id").head(nframes)

        # filter detections:
        tiny_detections = None
        if tracking_set.detections_gt is not None and not tracking_set.detections_gt.empty:
            tiny_detections = tracking_set.detections_gt[
                tracking_set.detections_gt.image_id.isin(tiny_image_metadatas.index)
            ]

        assert len(tiny_video_metadatas) > 0, "No videos left after subsampling the tracking set"
        assert len(tiny_image_metadatas) > 0, "No images left after subsampling the tracking set"

        return TrackingSet(
            tiny_video_metadatas,
            tiny_image_metadatas,
            tiny_detections,
            tiny_image_gt,
        )
