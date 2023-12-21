from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
from pbtrack.utils.cv2 import cv2_load_image, crop_bbox_ltwh
from pbtrack.utils.easyocr import bbox_easyocr_to_image_ltwh

from pbtrack.pipeline.videolevel_module import VideoLevelModule
from pbtrack.utils.openmmlab import get_checkpoint

import easyocr

import logging

log = logging.getLogger(__name__)


class VotingTrackletJurseyNumber(VideoLevelModule):
    
    input_columns = []
    output_columns = ["jn_number_tracklet"]
    
    def __init__(self, cfg, device, tracking_dataset=None):
        super().__init__(batch_size=1)

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        print("TrackletJurseyNumber process")
        print("detections", detections.head())
        for col in detections.columns:
            print(col)
        
        return detections
