import os
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from tracklab.datastruct import TrackingDataset, TrackingSet

log = logging.getLogger(__name__)


class MOT(TrackingDataset):
    def __init__(self, dataset_path: str, categories_list: list, nvid: int = -1,
                 vids_dict: list = None, public_dets_subpath : str = None, *args, **kwargs):
        self.categories_list = categories_list
        self.dataset_path = Path(dataset_path)
        self.public_dets_subpath = public_dets_subpath
        assert self.dataset_path.exists(), "'{}' directory does not exist".format(
            self.dataset_path
        )

        sets_dict = {}
        for set in ['train', 'val', 'test']:
            set_path = self.dataset_path / set
            if os.path.isdir(set_path):
                sets_dict[set] = self.load_set(set_path, nvid, vids_dict[set])
            else:
                log.warning(f"Warning: the {set} set does not exist.")
                sets_dict[set] = None
        super().__init__(dataset_path, sets_dict, *args, **kwargs)


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
            df['bbox_conf'] = 1/(1 + np.exp(-df['bbox_conf']))
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
        video_list.sort()

        if nvid > 0:
            video_list = video_list[:nvid]

        if vids_filter_set is not None and len(vids_filter_set) > 0:
            missing_videos = set(vids_filter_set) - set(video_list)
            if missing_videos:
                log.warning(
                    f"Warning: The following videos provided in config 'dataset.vids_dict' do not exist in {split} set: {missing_videos}")

            video_list = [video for video in video_list if video in vids_filter_set]

        image_counter = 0
        person_counter = 0
        for video_folder in tqdm(sorted(video_list), desc=f"Loading MOT20 '{split}' set videos"):  # Sort videos by name
            video_folder_path = os.path.join(dataset_path, video_folder)
            if os.path.isdir(video_folder_path):
                # Read seqinfo.ini
                seqinfo_path = os.path.join(video_folder_path, 'seqinfo.ini')
                seqinfo_data = self.read_ini_file(seqinfo_path)

                # Read ground truth detections
                gt_path = os.path.join(video_folder_path, 'gt', 'gt.txt')
                if os.path.isfile(gt_path):
                    detections_df = self.read_motchallenge_formatted_file(gt_path)
                    detections_df['person_id'] = detections_df['track_id'] - 1 + person_counter
                    detections_df['image_id'] = detections_df['image_id'] - 1 + image_counter
                    detections_df['video_id'] = len(video_metadatas_list) + 1
                    # detections_df['visibility'] = 1  # FIXME not sure to put it to 1
                    detections_list.append(detections_df)
                    person_counter += len(detections_df['track_id'].unique())
                else:
                    log.warning(
                        f"Warning: The {video_folder} from {split} split does not contain ground truth.")

                # read public detections file
                if self.public_dets_subpath is not None:
                    det_path = os.path.join(video_folder_path, self.public_dets_subpath)
                    if os.path.isfile(det_path):
                        detections_df = self.read_motchallenge_result_formatted_file(det_path)
                        detections_df['image_id'] = detections_df['image_id'] - 1 + image_counter
                        detections_df['video_id'] = len(video_metadatas_list) + 1
                        public_detections_list.append(detections_df)
                    else:
                        log.warning(
                            f"Warning: The {video_folder} from {split} split does not contain public detections.")

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
                columns=['image_id', 'track_id', 'bbox_ltwh', 'bbox_conf', 'class', 'visibility', 'person_id', 'video_id'])
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
        detections['id'] = detections['video_id'].astype(str) + "_" + detections['image_id'].astype(str) + "_" + detections[
            'track_id'].astype(str)

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

        tracking_set = TrackingSet(
            video_metadata,
            image_metadata,
            detections,
            image_gt,
        )
        if self.public_dets_subpath is not None:
            tracking_set.detections_public = public_detections

        return tracking_set
