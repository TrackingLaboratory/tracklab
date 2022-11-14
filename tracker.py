from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
import os
import json

class Source(Enum):
    METADATA = 0 # just an image
    DET = 1 # model detection
    TRACK = 2 # model tracking

@dataclass
class Metadata:
    file_path: str # path/video_name/file.jpg
    height: int
    width: int
    image_id: Optional[str] = None # for eval of posetrack
    file_name: Optional[str] = None # file.jpg
    video_name: Optional[str] = None # video_name
    frame: Optional[int] = None
    nframes: Optional[int] = None

@dataclass
class Bbox:
    x: float
    y: float
    w: float
    h: float
    conf: float

@dataclass
class Keypoint:
    x: float
    y: float
    conf: float
    part: int

@dataclass
class Detection:
    metadata: Optional[Metadata] = None
    source: Optional[Source] = 0
    bbox: Optional[Bbox] = None
    keypoints: Optional[list[Keypoint]] = None
    reid_features : Optional[np.ndarray] = None
    person_id: Optional[int] = -1

    def asdict(self): 
        keypoints = {}
        if self.keypoints:
            for i, keypoint in enumerate(self.keypoints):
                keypoints = {**keypoints,
                **{f"kp{i}_{k}":v for k,v in asdict(keypoint).items()},
                }
        bboxes = {}
        if self.bbox:
            bboxes = {f"bb_{k}":v for k,v in asdict(self.bbox).items()}
        return {
            'source': self.source,
            'person_id': self.person_id,
            **asdict(self.metadata),
            **bboxes,
            **keypoints,
            }
    
    # TODO change location of those functions
    def rescale_xy(self, coords, input_shape, output_shape=None):
        if output_shape is None:
            output_shape = (self.metadata.height, self.metadata.width)
        x_ratio = output_shape[1]/input_shape[1]
        y_ratio = output_shape[0]/input_shape[0]
        coords[:, 0] *= x_ratio
        coords[:, 1] *= y_ratio
        return coords
    
    def bbox_xyxy(self, image_shape=None):
        xyxy = [self.bbox.x, self.bbox.y, self.bbox.x+self.bbox.w, self.bbox.y+self.bbox.h]
        if image_shape is not None:
            x_ratio = image_shape[1]/self.metadata.width
            y_ratio = image_shape[0]/self.metadata.height
            xyxy[[0,2]] *= x_ratio
            xyxy[[1,3]] *= y_ratio
        return np.array(xyxy)
    
    def bbox_xywh(self, image_shape=None):
        xywh = [self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h]
        if image_shape is not None:
            x_ratio = image_shape[1]/self.metadata.width
            y_ratio = image_shape[0]/self.metadata.height
            xywh[[0,2]] *= x_ratio
            xywh[[1,3]] *= y_ratio
        return np.array(xywh)

class Tracker(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(Tracker, self).__init__(*args, **kwargs)
    
    @property
    def _constructor(self):
        return Tracker

    @property
    def _constructor_sliced(self):
        return TrackerSeries
    
    def bbox_xywh(self, with_conf=False):
        return self[['bb_x', 'bb_y', 'bb_w', 'bb_h']].values
        if with_conf:
            return self[['bb_x', 'bb_y', 'bb_w', 'bb_h', 'bb_conf']].values
        else:
            return self[['bb_x', 'bb_y', 'bb_w', 'bb_h']].values
    
    def bbox_xyxy(self, with_conf=False):
        self['bb_x2'] = self.apply(lambda row: row.bb_x+row.bb_w, axis=1)
        self['bb_y2'] = self.apply(lambda row: row.bb_y+row.bb_h, axis=1)
        if with_conf:
            return self[['bb_x', 'bb_y', 'bb_x2', 'bb_y2', 'bb_conf']].values
        else:
            return self[['bb_x', 'bb_y', 'bb_x2', 'bb_y2']].values
    
    def pose_xy(self, with_conf=False):
        if with_conf:
            return self.loc[:, self.columns.str.endswith(('x', 'y', 'conf')) & \
                       self.columns.str.startswith('kp')].values.reshape((-1, 17, 3))
        else:
            return self.loc[:, self.columns.str.endswith(('x', 'y')) & \
                       self.columns.str.startswith('kp')].values.reshape((-1, 17, 2))
            
    def save_mot(self, path):
        videos = self.video_name.unique()
        for video in videos:
            output = ''
            sub_df = self[(self['video_name'] == video) & (self['source'] == 2)].sort_values(by=['frame'])
            for index, detection in sub_df.iterrows():
                output += f"{detection['frame']}, {detection['person_id']}, {detection['bb_x']}, " + \
                        f"{detection['bb_y']}, {detection['bb_w']}, " + \
                        f"{detection['bb_h']}, {detection['bb_conf']}, -1, -1, -1\n"
            file_name = os.path.join(path, video + '.txt')
            with open(file_name, 'w+') as f:
                f.write(output)
        
    # FIXME maybe merge with save_pose_tracking ?
    def save_pose_estimation(self, path):
        videos = self.video_name.unique()
        for video in videos:
            video_df = self[(self['video_name'] == video)].sort_values(by=['frame'])
            detection_df = video_df[video_df['source'] >= 1]
            
            keypoints = detection_df.pose_xy(with_conf=True)
            annotations = []
            for ((index, detection), xys) in zip(detection_df.iterrows(), keypoints):
                annotations.append({
                    'bbox': [detection['bb_x'], detection['bb_y'], 
                             detection['bb_w'], detection['bb_h']],
                    'image_id': detection['image_id'],
                    'keypoints': xys, # FIXME score -> visibility
                    'scores': xys[:,2],
                    'person_id': index, # This is a dummy variable
                    'track_id': index, # This is a dummy variable
                })
            
            file_paths = video_df['file_path'].unique()
            image_ids = video_df['image_id'].unique()
            images = []
            for file_path, image_id in zip(file_paths, image_ids):
                images.append({
                    'file_name': file_path,
                    'id': image_id,
                    'image_id': image_id,
                })
            
            file_name = os.path.join(path, video + '.json')
            with open(file_name, 'w+') as f:
                output = {
                    'images': images,
                    'annotations': annotations,
                }
                json.dump(output, f, cls=CustomEncoder)
    
    def save_pose_tracking(self, path):
        videos = self.video_name.unique()
        for video in videos:
            video_df = self[(self['video_name'] == video)].sort_values(by=['frame'])
            detection_df = video_df[video_df['source'] == 2]
            
            keypoints = detection_df.pose_xy(with_conf=True)
            annotations = []
            for ((_, detection), xys) in zip(detection_df.iterrows(), keypoints):
                annotations.append({
                    'bbox': [detection['bb_x'], detection['bb_y'], 
                             detection['bb_w'], detection['bb_h']],
                    'image_id': detection['image_id'],
                    'keypoints': xys, # FIXME score -> visibility
                    'scores': xys[:,2],
                    'person_id': detection['person_id'],
                    'track_id': detection['person_id'],
                })
            
            file_paths = video_df['file_path'].unique()
            image_ids = video_df['image_id'].unique()
            images = []
            for file_path, image_id in zip(file_paths, image_ids):
                images.append({
                    'file_name': file_path,
                    'id': image_id,
                    'image_id': image_id,
                })
            
            file_name = os.path.join(path, video + '.json')
            with open(file_name, 'w+') as f:
                output = {
                    'images': images,
                    'annotations': annotations,
                }
                json.dump(output, f, cls=CustomEncoder)
    
class TrackerSeries(pd.Series):
    @property
    def _constructor(self):
        return TrackerSeries

    @property
    def _constructor_expanddim(self):
        return Tracker

# FIXME change my location
class CustomEncoder(json.JSONEncoder):
    """ 
    Special json encoder for numpy and pandas types 
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.isnan(obj) or pd.isna(obj):
            return -1
        else:
            return json.JSONEncoder.default(self, obj)