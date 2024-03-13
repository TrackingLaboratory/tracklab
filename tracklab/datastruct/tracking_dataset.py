import logging
import os
from abc import ABC
from pathlib import Path
from dataclasses import dataclass
import pandas as pd


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
        self.sets = SetsDict(sets)
        sub_sampled_sets = SetsDict()
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
            if file_df["frame"].min() == 0:
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
        return results
