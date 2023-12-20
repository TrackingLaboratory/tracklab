from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
from pbtrack.utils.cv2 import cv2_load_image, crop_bbox_ltwh
from pbtrack.utils.easyocr import bbox_easyocr_to_image_ltwh

from pbtrack.pipeline.detectionlevel_module import DetectionLevelModule
from pbtrack.utils.openmmlab import get_checkpoint

import easyocr

import logging

log = logging.getLogger(__name__)


class EasyOCR(DetectionLevelModule):
    
    input_columns = []
    output_columns = ["jn_bbox_ltwh", "jn_number", "jn_confidence"]
    
    def __init__(self, cfg, device, tracking_dataset=None):
        super().__init__(batch_size=1)
        self.reader = easyocr.Reader(['en'])
        self.cfg = cfg

    # def __init__(self, cfg, device, batch_size):
    #     super().__init__(cfg, device, batch_size)
    #     self.reader = easyocr.Reader(['en'])
    #     self.cfg = cfg
    
    def no_jursey_number(self):
        return [None, None, 0]

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        # data = {
        #     "bbox": detection.bbox_ltwh,
        #     "file_path": metadata.file_path,
        #     "bbox_score": detection.bbox_conf,
        #     "bbox_id": 0,
        # }
        # return data
    
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        batch = {
            "img": crop,
            "bbox": detection.bbox_ltwh,
        }
        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jn_bbox_ltwh = []
        jn_number = []
        jn_confidence = []
        # for file_path, bbox in zip(batch['file_path'], batch['bbox']):
            # img = cv2_load_image(file_path)
            # img = crop_bbox_ltwh(img, bbox)
        for img, bbox in zip(batch['img'], batch['bbox']):   
            img_np = img.cpu().numpy()
            # bbox_np = bbox.cpu().numpy()
            bbox = bbox.cpu()
            result = self.reader.readtext(img_np, **self.cfg)   
            if result == []:
                jn = self.no_jursey_number()
            else:
                result = result[0] # only take the first result (highest confidence)
                try:
                    # see if the result is a number
                    int(result[1])
                except ValueError:
                    jn = self.no_jursey_number()
                else:
                    jn = [bbox_easyocr_to_image_ltwh(result[0], bbox), result[1], result[2]]
                
            jn_bbox_ltwh.append(jn[0])
            jn_number.append(jn[1])
            jn_confidence.append(jn[2])
        detections['jn_bbox_ltwh'] = jn_bbox_ltwh
        detections['jn_number'] = jn_number
        detections['jn_confidence'] = jn_confidence
        
        return detections
