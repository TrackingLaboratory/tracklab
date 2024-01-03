import os
import pandas as pd
from pathlib import Path
from tracklab.datastruct import TrackingDataset, TrackingSet


class SoccerNetMOT(TrackingDataset):
    def __init__(self, dataset_path: str, *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), f"'{self.dataset_path}' directory does not exist. Please check the path or download the dataset following the instructions here: https://github.com/SoccerNet/sn-tracking"

        train_set = load_set(self.dataset_path / "train")
        val_set = load_set(self.dataset_path / "test")
        # test_set = load_set(self.dataset_path / "challenge")
        test_set = None

        super().__init__(dataset_path, train_set, val_set, test_set, *args, **kwargs)


def read_ini_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return dict(line.strip().split('=') for line in lines[1:])


def read_motchallenge_formatted_file(file_path):
    columns = ['image_id', 'track_id', 'left', 'top', 'width', 'height', 'confidence', 'class', 'visibility', 'unused']
    df = pd.read_csv(file_path, header=None, names=columns)
    df['bbox_ltwh'] = df.apply(lambda row: [row['left'], row['top'], row['width'], row['height']], axis=1)
    df['person_id'] = df['track_id']  # Create person_id column with the same content as track_id
    return df[['image_id', 'track_id', 'person_id', 'bbox_ltwh', 'class', 'visibility']]


def load_set(dataset_path):
    video_metadatas_list = []
    image_metadata_list = []
    detections_list = []
    categories_list = []

    for video_folder in sorted(os.listdir(dataset_path)):  # Sort videos by name
        video_folder_path = os.path.join(dataset_path, video_folder)
        if os.path.isdir(video_folder_path):
            # Read gameinfo.ini
            gameinfo_path = os.path.join(video_folder_path, 'gameinfo.ini')
            gameinfo_data = read_ini_file(gameinfo_path)

            # Read seqinfo.ini
            seqinfo_path = os.path.join(video_folder_path, 'seqinfo.ini')
            seqinfo_data = read_ini_file(seqinfo_path)

            # Read ground truth detections
            gt_path = os.path.join(video_folder_path, 'gt', 'gt.txt')
            detections_df = read_motchallenge_formatted_file(gt_path)
            detections_df['video_id'] = len(video_metadatas_list) + 1
            detections_list.append(detections_df)

            # Append video metadata
            video_metadata = {
                'id': len(video_metadatas_list) + 1,
                'name': gameinfo_data.get('name', ''),
                'nframes': int(seqinfo_data.get('seqLength', 0)),
                'frame_rate': int(seqinfo_data.get('frameRate', 0)),
                'seq_length': int(seqinfo_data.get('seqLength', 0)),
                'im_width': int(seqinfo_data.get('imWidth', 0)),
                'im_height': int(seqinfo_data.get('imHeight', 0)),
                'game_id': int(gameinfo_data.get('gameID', 0)),
                'action_position': int(gameinfo_data.get('actionPosition', 0)),
                'action_class': gameinfo_data.get('actionClass', ''),
                'visibility': gameinfo_data.get('visibility', ''),
                'clip_start': int(gameinfo_data.get('clipStart', 0)),
                'game_time_start': gameinfo_data.get('gameTimeStart', '').split(' - ')[1],
                # Remove the half period index
                'game_time_stop': gameinfo_data.get('gameTimeStop', '').split(' - ')[1],  # Remove the half period index
                'clip_stop': int(gameinfo_data.get('clipStop', 0)),
                'num_tracklets': int(gameinfo_data.get('num_tracklets', 0)),
                'half_period_start': int(gameinfo_data.get('gameTimeStart', '').split(' - ')[0]),
                # Add the half period start column
                'half_period_stop': int(gameinfo_data.get('gameTimeStop', '').split(' - ')[0]),
                # Add the half period stop column
            }

            # Extract categories from trackletID entries
            for i in range(1, int(gameinfo_data.get('num_tracklets', 0)) + 1):
                tracklet_entry = gameinfo_data.get(f'trackletID_{i}', '')
                category, position = tracklet_entry.split(';')
                class_name = f"{category.strip().replace(' ', '_')}_{position.replace(' ', '_')}"  # fixme class name is not unique accross videos
                categories_list.append({'supercategory': 'person', 'id': i, 'name': class_name})

            video_metadata['categories'] = categories_list  # Add categories to the video metadata

            # Append video metadata
            video_metadatas_list.append(video_metadata)

            # Append image metadata
            img_folder_path = os.path.join(video_folder_path, 'img1')
            img_metadata_df = pd.DataFrame({
                'frame': [i for i in range(1, int(seqinfo_data.get('seqLength', 0)) + 1)],
                # 'id': [f'{len(video_metadatas_list)}_{i}' for i in range(1, int(seqinfo_data.get('seqLength', 0)) + 1)],
                'video_id': len(video_metadatas_list),
                'file_path': [os.path.join(img_folder_path, f'{i:06d}.jpg') for i in
                              range(1, int(seqinfo_data.get('seqLength', 0)) + 1)],

            })
            image_metadata_list.append(img_metadata_df)

    # Add categories for goalkeepers
    categories_list.append({'supercategory': 'person', 'id': len(categories_list) + 1, 'name': 'goalkeeper_team_left'})
    categories_list.append({'supercategory': 'person', 'id': len(categories_list) + 1, 'name': 'goalkeeper_team_right'})

    # Convert list to a set to remove duplicates
    categories_set = list({category['name']: category for category in categories_list}.values())

    # Sort the list by 'id' for consistent ordering
    categories_set = sorted(categories_set, key=lambda x: x['id'])

    # Assign the categories to the video metadata
    for video_metadata in video_metadatas_list:
        video_metadata['categories'] = categories_set

    # Concatenate dataframes
    video_metadata = pd.DataFrame(video_metadatas_list)
    image_metadata = pd.concat(image_metadata_list, ignore_index=True)
    detections = pd.concat(detections_list, ignore_index=True)

    # Set 'id' column as the index   in the detections and image dataframe
    detections['id'] = detections.index
    image_metadata['id'] = image_metadata.index

    detections.set_index("id", drop=True, inplace=True)
    image_metadata.set_index("id", drop=True, inplace=True)
    video_metadata.set_index("id", drop=True, inplace=True)

    # Reorder columns in dataframes
    video_metadata = video_metadata[
        ['name', 'nframes', 'frame_rate', 'seq_length', 'im_width', 'im_height', 'game_id', 'action_position',
         'action_class', 'visibility', 'clip_start', 'game_time_start', 'clip_stop', 'game_time_stop', 'num_tracklets',
         'half_period_start', 'half_period_stop', 'categories']]
    image_metadata = image_metadata[['video_id', 'frame', 'file_path']]
    detections = detections[['image_id', 'video_id', 'track_id', 'person_id', 'bbox_ltwh', 'class', 'visibility']]

    return TrackingSet(
        video_metadata,
        image_metadata,
        detections,
    )
