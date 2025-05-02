from typing import Any

import numpy as np
import pandas as pd
import cv2
from rtmlib import YOLOX

from tracklab.pipeline import ImageLevelModule


class YOLOXDetector(ImageLevelModule):
    input_columns = []
    output_columns = [
        "image_id", "video_id", "category_id", "bbox_ltwh", "bbox_conf"
    ]

    def __init__(self, onnx_model, batch_size, device, input_size=(640, 640)):
        super().__init__(batch_size=batch_size)
        self.device = device
        self.onnx_model = onnx_model
        self.input_size = input_size
        self.onnx_model = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
        self.model = YOLOX(self.onnx_model, input_size, device=device)

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        padded_img = np.ones(
            (self.input_size[0], self.input_size[1], 3),
            dtype=np.uint8) * 114
        ratio = min(self.input_size[0] / image.shape[0],
                    self.input_size[1] / image.shape[1])

        resized_img = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(image.shape[0] * ratio), int(image.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        return image, ratio

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, ratios = batch

        return self.model(batch)