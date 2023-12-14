from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
from pbtrack.utils.cv2 import cv2_load_image, crop_bbox_ltwh
from pbtrack.utils.easyocr import bbox_easyocr_to_image_ltwh

from pbtrack.pipeline import JNDetector
from pbtrack.utils.openmmlab import get_checkpoint

import easyocr

import logging

log = logging.getLogger(__name__)


class EasyOCR(JNDetector):

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        self.reader = easyocr.Reader(['en'])
        self.cfg = cfg

    @torch.no_grad()
    def preprocess(self, detection: pd.Series, metadata: pd.Series):
        # image = cv2.imread(metadata.file_path)  # BGR not RGB !
        data = {
            "bbox": detection.bbox_ltwh,
            "file_path": metadata.file_path,
            "bbox_score": detection.bbox_conf,
            "bbox_id": 0,
        }
        return data

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame):
        jn_bbox_ltwh = []
        jn_number = []
        jn_confidence = []
        for file_path, bbox in zip(batch['file_path'], batch['bbox']):
            img = cv2_load_image(file_path)
            img = crop_bbox_ltwh(img, bbox)
            
            result = self.reader.readtext(img, **self.cfg)   
            if result == []:
                result = [None, None, 0]
                jn_bbox_ltwh.append(result[0])
                jn_number.append(result[1])
                jn_confidence.append(result[2])
            else:
                result = result[0] # only take the first result (highest confidence)
                jn_bbox_ltwh.append(bbox_easyocr_to_image_ltwh(result[0], bbox))
                jn_number.append(int(result[1]))
                jn_confidence.append(result[2])
        detections['jn_bbox_ltwh'] = jn_bbox_ltwh
        detections['jn_number'] = jn_number
        detections['jn_confidence'] = jn_confidence
        
        return detections
