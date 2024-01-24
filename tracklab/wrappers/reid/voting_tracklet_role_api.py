import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from tracklab.utils.attribute_voting import select_highest_voted_att
from tracklab.pipeline.videolevel_module import VideoLevelModule
import logging


log = logging.getLogger(__name__)


class VotingTrackletRole(VideoLevelModule):
    
    input_columns = ["track_id", "roles"]
    output_columns = ["role_tracklet"]
    
    def __init__(self, cfg, device, tracking_dataset=None):
        pass
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        detections["role_tracklet"] = [np.nan] * len(detections)
        if "track_id" not in detections.columns:
            return detections
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            roles = tracklet.roles
            tracklet_role = [select_highest_voted_att(roles)] * len(tracklet)            
            detections.loc[tracklet.index, "role_tracklet"] = tracklet_role
        
        return detections
