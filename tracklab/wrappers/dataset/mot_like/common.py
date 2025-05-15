import copy
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from multiprocessing import Pool

from rich.progress import track
from pathlib import Path

from tracklab.datastruct import TrackingDataset, TrackingSet
from tracklab.utils import wandb

log = logging.getLogger(__name__)


class MOT(TrackingDataset):
    def __init__(self, dataset_path: str, categories_list: list, nvid: int = -1, nframes: int = -1,
                 vids_dict: list = None, public_dets_subpath: str = None,
                 leave_one_out_idx: Optional[int] = None, *args, **kwargs):
        self.categories_list = categories_list
        self.dataset_path = Path(dataset_path)
        self.public_dets_subpath = public_dets_subpath
        assert self.dataset_path.exists(), "'{}' directory does not exist".format(
            self.dataset_path
        )

        set_names = ['train', 'val', 'test']
        with Pool(processes=3) as pool:
            pool_args = [(set_name, self.dataset_path, nvid, vids_dict[set_name]) for set_name in set_names]
            results = pool.map(self.load_set_wrapper, pool_args)

        sets_dict = {set_name: result for set_name, result in results}
        if leave_one_out_idx is not None:
            video_ids = list(sets_dict["train"].video_metadatas.id)
            loo_video_id = int(sets_dict["train"].video_metadatas.iloc[leave_one_out_idx].id)
            video_ids.remove(loo_video_id)
            train_set = copy.deepcopy(sets_dict["train"])
            val_set = copy.deepcopy(sets_dict["train"])
            train_set.filter_videos(video_ids)
            sets_dict["train"] = train_set
            val_set.filter_videos([loo_video_id])
            sets_dict["val"] = val_set
        log.info(sets_dict.keys())
        super().__init__(dataset_path, sets_dict, nvid, nframes, vids_dict, *args, **kwargs)

    def load_set_wrapper(self, args):
        set_name, dataset_path, nvid, vids_dict = args
        set_path = dataset_path / set_name
        if os.path.isdir(set_path):
            return set_name, self.load_set(set_path, nvid, vids_dict)
        else:
            log.warning(f"The {set_name} split does not exist.")
            return set_name, None

    def read_ini_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return {k: v for line in lines if len((split_line := line.strip().split('='))) == 2 for k, v in [split_line]}

    def read_motchallenge_formatted_file(self, file_path):
        columns = ['image_id', 'track_id', 'left', 'top', 'width', 'height', 'bbox_conf', 'class', 'visibility']
        df = pd.read_csv(file_path, header=None, names=columns)
        df['bbox_ltwh'] = df.apply(lambda row: np.array([row['left'], row['top'], row['width'], row['height']]), axis=1)
        return df[['image_id', 'track_id', 'bbox_ltwh', 'bbox_conf', 'class', 'visibility']]

    def read_motchallenge_result_formatted_file(self, file_path):
        columns = ['image_id', 'track_id', 'left', 'top', 'width', 'height', 'bbox_conf', 'x', 'y', 'z']
        df = pd.read_csv(file_path, header=None, names=columns)
        df['bbox_ltwh'] = df.apply(lambda row: np.array([row['left'], row['top'], row['width'], row['height']]), axis=1)
        df['category_id'] = 1
        if df['bbox_conf'].max() > 1 and df['bbox_conf'].min() < 0:
            log.warning("Warning: 'bbox_conf' from the public detections is not between 0 and 1.")
            df['bbox_conf'] = 1 / (1 + np.exp(-df['bbox_conf']))
        elif df['bbox_conf'].max() < 0:
            log.warning("Warning: 'bbox_conf' from the public detections is not between 0 and 1.")
            df['bbox_conf'] = 1.0
        return df[['image_id', 'bbox_ltwh', 'bbox_conf', 'category_id']]

    def load_set(self, dataset_path, nvid=-1, vids_filter_set=None):
        video_metadatas_list = []
        image_metadata_list = []
        detections_list = []
        public_detections_list = []
        split = os.path.basename(dataset_path)  # Get the split name from the dataset path
        video_list = os.listdir(dataset_path)
        video_list = [v for v in video_list if not v.startswith('.')]
        video_list.sort()

        if nvid > 0:
            assert vids_filter_set is None or len(vids_filter_set) == 0, "Cannot use both 'nvid' and 'vids_dict' arguments at the same time."
            video_list = video_list[:nvid]

        image_counter = 0
        person_counter = 0
        warning_trigger_gt = False
        warning_trigger_public_det = False
        for video_folder in track(sorted(video_list),
                                 f"Loading {self.__class__.__name__} '{split}' set videos"):  # Sort videos by name
            video_folder_path = os.path.join(dataset_path, video_folder)
            if os.path.isdir(video_folder_path):
                # Read seqinfo.ini
                seqinfo_path = os.path.join(video_folder_path, 'seqinfo.ini')
                seqinfo_data = self.read_ini_file(seqinfo_path)

                # Read ground truth detections
                gt_path = os.path.join(video_folder_path, 'gt', 'gt.txt')
                if os.path.isfile(gt_path):
                    detections_df = self.read_motchallenge_formatted_file(gt_path)
                    detections_df['person_id'] = detections_df['track_id'] + person_counter
                    detections_df['image_id'] = detections_df['image_id'] - 1 + image_counter
                    detections_df['video_id'] = len(video_metadatas_list) + 1
                    # detections_df['visibility'] = 1  # FIXME not sure to put it to 1
                    detections_list.append(detections_df)
                    person_counter += len(detections_df['track_id'].unique())
                else:
                    if not warning_trigger_gt:
                        warning_trigger_gt = True
                        log.warning(f"Warning: The {split} split does not contain ground truth.")

                # read public detections file
                if self.public_dets_subpath is not None:
                    det_path = os.path.join(video_folder_path, self.public_dets_subpath)
                    if os.path.isfile(det_path):
                        detections_df = self.read_motchallenge_result_formatted_file(det_path)
                        if detections_df['image_id'].min() == 1:
                            detections_df['image_id'] = detections_df['image_id'] - 1
                        detections_df['image_id'] = detections_df['image_id'] + image_counter
                        detections_df['video_id'] = len(video_metadatas_list) + 1
                        public_detections_list.append(detections_df)
                    else:
                        if not warning_trigger_public_det:
                            warning_trigger_public_det = True
                            log.warning(f"Warning: The {split} split does not contain public detections.")

                # Append video metadata
                nframes = int(seqinfo_data.get('seqLength', 0))
                video_metadata = {
                    'id': len(video_metadatas_list) + 1,
                    'nframes': nframes,
                    'frame_rate': int(seqinfo_data.get('frameRate', 0)),
                    'seq_length': nframes,
                    'im_width': int(seqinfo_data.get('imWidth', 0)),
                    'im_height': int(seqinfo_data.get('imHeight', 0)),
                    'name': video_folder,
                }

                # Append video metadata
                video_metadatas_list.append(video_metadata)

                # Append image metadata
                img_folder_path = os.path.join(video_folder_path, 'img1')
                img_metadata_df = pd.DataFrame({
                    'frame': [i for i in range(0, nframes)],
                    'nframes': nframes,
                    'id': [image_counter + i for i in range(0, nframes)],
                    'video_id': len(video_metadatas_list),
                    'file_path': [os.path.join(img_folder_path, f) for f in
                                  sorted([f for f in os.listdir(img_folder_path) if f.endswith('.jpg')])],
                })
                image_counter += nframes
                image_metadata_list.append(img_metadata_df)

        # Assign the categories to the video metadata  # TODO at dataset level?
        for video_metadata in video_metadatas_list:
            video_metadata['categories'] = self.categories_list

        # Concatenate dataframes
        video_metadata = pd.DataFrame(video_metadatas_list)
        image_metadata = pd.concat(image_metadata_list, ignore_index=True)
        if len(detections_list):
            detections = pd.concat(detections_list, ignore_index=True)
        else:
            detections = pd.DataFrame(
                columns=['image_id', 'track_id', 'bbox_ltwh', 'bbox_conf', 'class', 'visibility', 'person_id',
                         'video_id'])
        if self.public_dets_subpath is not None:
            if len(public_detections_list):
                public_detections = pd.concat(public_detections_list, ignore_index=True)
                public_detections = public_detections.sort_values(by=['video_id', 'image_id'],
                                                                  ascending=[True, True])
            else:
                public_detections = pd.DataFrame(
                    columns=['image_id', 'bbox_ltwh', 'bbox_conf', 'video_id', 'category_id'])

        # Use video_id, image_id, track_id as unique id
        detections = detections.sort_values(by=['video_id', 'image_id', 'track_id'], ascending=[True, True, True])
        # detections['id'] = detections['video_id'].astype(str) + "_" + \
        #    detections['image_id'].astype(str) + "_" + detections['track_id'].astype(str)
        detections['id'] = detections.index

        # Add category id to detections
        detections['category_id'] = detections['class']

        detections.set_index("id", drop=False, inplace=True)
        image_metadata.set_index("id", drop=False, inplace=True)
        video_metadata.set_index("id", drop=False, inplace=True)

        # Add is_labeled column to image_metadata
        image_metadata['is_labeled'] = True

        # Reorder columns in dataframes
        video_metadata_columns = ['name', 'nframes', 'frame_rate', 'seq_length', 'im_width', 'im_height']
        video_metadata_columns.extend(set(video_metadata.columns) - set(video_metadata_columns))
        video_metadata = video_metadata[video_metadata_columns]
        image_metadata_columns = ['video_id', 'frame', 'file_path', 'is_labeled']
        image_metadata_columns.extend(set(image_metadata.columns) - set(image_metadata_columns))
        image_metadata = image_metadata[image_metadata_columns]
        image_gt = image_metadata.copy()
        detections_column_ordered = ['image_id', 'video_id', 'track_id', 'person_id', 'bbox_ltwh', 'bbox_conf', 'class',
                                     'visibility']
        detections_column_ordered.extend(set(detections.columns) - set(detections_column_ordered))
        detections = detections[detections_column_ordered]

        # filter out videos not in vids_filter_set
        # FIXME should normally be done before loading the videos, but cannot do that because it would change some ids from one run to the other, since image_id, person_id etc are computed with a counter.
        if vids_filter_set is not None and len(vids_filter_set) > 0:
            missing_videos = set(vids_filter_set) - set(video_list)
            assert not missing_videos, f"The following videos provided in config 'dataset.vids_dict' do not exist in {split} set: {missing_videos}"
            video_list = [video for video in video_list if video in vids_filter_set]
            # get video ids
            video_ids = set(video_metadata[video_metadata['name'].isin(video_list)]['id'].tolist())
            # filter out detections, image_metadata and video_metadata
            detections = detections[detections['video_id'].isin(video_ids)]
            image_metadata = image_metadata[image_metadata['video_id'].isin(video_ids)]
            video_metadata = video_metadata[video_metadata['id'].isin(video_ids)]

        tracking_set = TrackingSet(
            video_metadata,
            image_metadata,
            detections,
            image_gt,
        )
        if self.public_dets_subpath is not None:
            tracking_set.detections_public = public_detections

        return tracking_set

    def process_trackeval_results(self, results, dataset_config, eval_config):
        if "SUMMARIES" in results and "pedestrian" in results["SUMMARIES"]:
            res = {
                f"{k}": float(v) if '.' in v else int(v)
                for _, metrics in results["SUMMARIES"]["pedestrian"].items()
                for k, v in metrics.items()
            }
            wandb.log(res)

        res_by_vid = {}
        for video_name, video_data in results.items():
            if video_name != "SUMMARIES":
                for category, metrics in video_data["pedestrian"].items():
                    for metric_name, metric_value in metrics.items():
                        if not isinstance(metric_value, np.ndarray):  # Ignore np.array values
                            res_by_vid[f"tracking_by_video/{video_name}/{metric_name}"] = metric_value
        wandb.log(res_by_vid)
