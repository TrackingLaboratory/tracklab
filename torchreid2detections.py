from pathlib import Path

import numpy as np
import torch

from strong_sort.reid_multibackend import ReIDDetectMultiBackend


@torch.no_grad()
class Torchreid2detections():
    def __init__(self, device):
        reid_weights = 'reid/weights/osnet_x0_25_msmt17.pt'
        reid_weights = Path(reid_weights).resolve()
        self.model = ReIDDetectMultiBackend(weights=reid_weights, device=device, fp16=False)

    def _image2input(self, image): # Tensor RGB (1, 3, H, W)
        assert 1 == image.shape[0], "Test batch size should be 1"
        input = image[0].cpu().numpy() # -> (3, H, W)
        input = np.transpose(input, (1, 2, 0)) # -> (H, W, 3)
        input = input*255.0
        input = input.astype(np.uint8) # -> to uint8
        return input

    def run(self, detections, image):
        im_crops = []
        image = self._image2input(image)
        for det in detections.bboxes:
            box = det
            height, width = image.shape[:2]
            x1, y1, x2, y2 = self._xywh_to_xyxy(box, width, height)
            crop = image[y1:y2, x1:x2]
            im_crops.append(crop)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        detections.add_reid_features(features)
        return detections

    def _xywh_to_xyxy(self, bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2
