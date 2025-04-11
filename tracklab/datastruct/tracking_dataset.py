import copy
import logging
import os
from abc import ABC
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from tracklab.utils import wandb

log = logging.getLogger(__name__)


class SetsDict(dict):
    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f"Trying to access a '{key}' split of the dataset that is not available. "
                           f"Available splits are {list(self.keys())}. "
                           f"Make sur this split name is correct or is available in the dataset folder.")
        return super().__getitem__(key)


@dataclass
class TrackingSet:
    video_metadatas: pd.DataFrame
    image_metadatas: pd.DataFrame
    detections_gt: pd.DataFrame
    image_gt: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["video_id"]))

    def filter_videos(self, keep_video_ids):
        self.video_metadatas = self.video_metadatas.loc[keep_video_ids]
        self.image_metadatas = self.image_metadatas[self.image_metadatas.video_id.isin(keep_video_ids)]
        self.detections_gt = self.detections_gt[self.detections_gt.video_id.isin(keep_video_ids)]
        self.image_gt = self.image_gt[self.image_gt.video_id.isin(keep_video_ids)]


class TrackingDataset(ABC):
    def __init__(
        self,
        dataset_path: str,
        sets: dict[str, TrackingSet],
        nvid: int = -1,
        nframes: int = -1,
        vids_dict: list = None,
        *,
        set_split_idxs: Optional[dict[str, int]] = None,
        **kwargs
    ):
        set_split_idxs = set_split_idxs or {}
        self.dataset_path = Path(dataset_path)
        self.sets = SetsDict(sets)
        sub_sampled_sets = SetsDict()
        for set_name, split in self.sets.items():
            vid_list = vids_dict[set_name] if vids_dict is not None and set_name in vids_dict else None
            sub_sampled_sets[set_name] = self._subsample(split, nvid, nframes, vid_list)
        assert (len(set_split_idxs) == 0) or (nvid == -1), "Splitting the dataset and setting nvid to a different value is not supported"
        self.training_sets = copy.deepcopy(self.sets)
        self.sets = sub_sampled_sets
        self.set_splits = {}
        self.set_split_idxs = set_split_idxs

        for set_name, split_idx in set_split_idxs.items():
            self.set_splits[set_name] = []
            self._split_set(set_name)
            self.sets[set_name] = self.set_splits[set_name][split_idx]
            self.training_sets[set_name] = self.set_splits[set_name][split_idx]

    def _split_set(self, set_name, num_splits=2):
        video_groups = [[] for i in range(num_splits)]
        people_in_video = [set() for i in range(num_splits)]
        for video_id, _ in self.sets[set_name].detections_gt.groupby("video_id").person_id.nunique().sort_values(ascending=False).items():
            video_df = self.sets[set_name].detections_gt.loc[self.sets[set_name].detections_gt.video_id==video_id]
            for person_id in np.unique(video_df.person_id):
                group_idxs = np.nonzero([np.isin(person_id, x) for x in people_in_video])[0]
                if len(group_idxs) > 0:
                    current_group = group_idxs[0]
                    break
            else:
                current_group = np.argmin([len(x) for x in video_groups])  # group to put it in

            video_groups[current_group].append(video_id)
            people_in_video[current_group].update(video_df.person_id)

        self.train_sets = []
        for video_ids in video_groups:
            current_set = copy.deepcopy(self.sets[set_name])
            current_set.filter_videos(video_ids)
            self.set_splits[set_name].append(current_set)

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

        tiny_tracking_set = TrackingSet(
            tiny_video_metadatas,
            tiny_image_metadatas,
            tiny_detections,
            tiny_image_gt,
        )

        if hasattr(tracking_set, "detections_public") and not tracking_set.detections_public.empty:
            tiny_public_detections = tracking_set.detections_public[
                tracking_set.detections_public.image_id.isin(tiny_image_metadatas.index)
            ]
            tiny_tracking_set.detections_public = tiny_public_detections

        if hasattr(tracking_set, "detections_pred") and not tracking_set.detections_pred.empty:
            tiny_pred_detections = tracking_set.detections_pred[
                tracking_set.detections_pred.image_id.isin(tiny_image_metadatas.index)
            ]
            tiny_tracking_set.detections_pred = tiny_pred_detections

        return tiny_tracking_set


    @staticmethod
    def _mot_encoding(detections, image_metadatas, video_metadatas, bbox_column):
        detections = detections.copy()
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            detections.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
            suffixes=('', '_y')
        )
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "frame",
                "track_id",
                bbox_column,
            ],
            how="any",
            inplace=True,
        )

        if len_before_drop != len(df):
            log.warning(
                "Dropped {} rows with NA values".format(len_before_drop - len(df))
            )
        df["track_id"] = df["track_id"].astype(int)
        df["bb_left"] = df[bbox_column].apply(lambda x: x[0])
        df["bb_top"] = df[bbox_column].apply(lambda x: x[1])
        df["bb_width"] = df[bbox_column].apply(lambda x: x[2])
        df["bb_height"] = df[bbox_column].apply(lambda x: x[3])
        df = df.assign(x=-1, y=-1, z=-1)
        return df


    def save_for_eval(self,
                      detections: pd.DataFrame,
                      image_metadatas: pd.DataFrame,
                      video_metadatas: pd.DataFrame,
                      save_folder: str,
                      bbox_column_for_eval="bbox_ltwh",
                      save_classes=False,
                      is_ground_truth=False,
                      save_zip=True
                      ):
        """Save predictions in MOT Challenge format."""
        mot_df = self._mot_encoding(detections, image_metadatas, video_metadatas, bbox_column_for_eval)

        save_path = os.path.join(save_folder)
        os.makedirs(save_path, exist_ok=True)

        # MOT Challenge format = <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        # videos_names = mot_df["video_name"].unique()
        for id, video in video_metadatas.iterrows():
            file_path = os.path.join(save_path, f"{video['name']}.txt")
            file_df = mot_df[mot_df["video_id"] == id].copy()
            #if file_df["frame"].min() == 0:  # FIXME changed to make bytetrackddsort work?
            file_df["frame"] = file_df["frame"] + 1  # MOT Challenge format starts at 1
            if not file_df.empty:
                file_df.sort_values(by="frame", inplace=True)
                clazz = "category_id" if save_classes else "x"
                file_df[
                    [
                        "frame",
                        "track_id",
                        "bb_left",
                        "bb_top",
                        "bb_width",
                        "bb_height",
                        "bbox_conf",
                        clazz,
                        "y",
                        "z",
                    ]
                ].to_csv(
                    file_path,
                    header=False,
                    index=False,
                )
            else:
                open(file_path, "w").close()

    def process_trackeval_results(self, results, dataset_config, eval_config):
        log.info(f"TrackEval results = {results}")
        wandb.log(results)

    def __str__(self):
        set_str = []
        for set_name, set_data in self.sets.items():
            if set_data is not None:
                set_str.append(f"{set_name} set: {len(set_data.video_metadatas)}")
        return self.__class__.__name__ + "= " + "; ".join(set_str)
