import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from tracklab.utils.attribute_voting import select_highest_voted_att
from tracklab.pipeline.videolevel_module import VideoLevelModule
import logging


log = logging.getLogger(__name__)


class TrackletSideLabeling(VideoLevelModule):
    
    input_columns = ["track_id", "team_tracklet", "bbox_ltwh"]
    output_columns = ["side_tracklet"]
    
    def __init__(self, cfg, device, tracking_dataset=None):
        pass
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        if "track_id" not in detections.columns:
            return detections
        
        team_a = detections[detections.team_tracklet == 0]
        team_b = detections[detections.team_tracklet == 1]
        ya_coordinates = [bbox[0] for bbox in team_a.bbox_ltwh] # (x, y) are the center of a bbox 
        yb_coordinates = [bbox[0] for bbox in team_b.bbox_ltwh] # (x, y) are the center of a bbox
        
        avg_a = sum(ya_coordinates) / len(ya_coordinates)
        avg_b = sum(yb_coordinates) / len(yb_coordinates)
        
        if avg_a > avg_b:           
            detections.loc[team_a.index, "side_tracklet"] = ['right'] * len(team_a)
            detections.loc[team_b.index, "side_tracklet"] = ['left'] * len(team_b)
        else:
            detections.loc[team_a.index, "side_tracklet"] = ['left'] * len(team_a)
            detections.loc[team_b.index, "side_tracklet"] = ['right'] * len(team_b)
            
        return detections