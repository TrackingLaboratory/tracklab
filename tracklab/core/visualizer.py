from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from tracklab.core.visualization_engine import prediction_cmap, ground_truth_cmap


class Visualizer(ABC):
    @abstractmethod
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        pass

    def post_init(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ImageVisualizer(Visualizer, ABC):
    pass


class DetectionVisualizer(Visualizer, ABC):
    def __init__(self):
        self.colors = None

    def post_init(self, colors, **kwargs):
        super().post_init(**kwargs)
        self.colors = colors

    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        bbox_pred = torch.tensor(np.stack(detections_pred.bbox.ltrb()))
        bbox_gt = torch.tensor(np.stack(detections_gt.bbox.ltrb()))
        cost_matrix = box_iou(bbox_pred, bbox_gt)

        row_idxs, col_idxs = linear_sum_assignment(1 - cost_matrix)
        gt_rest = set(range(len(bbox_gt))) - set(col_idxs)
        for i in range(max(len(bbox_pred), len(bbox_gt))):
            if i not in row_idxs:
                metric = None
                if len(bbox_pred) < len(bbox_gt):
                    pred = None
                    gt_id = gt_rest.pop()
                    gt = detections_gt.iloc[gt_id]
                else:
                    pred = detections_pred.iloc[i]
                    gt = None
            else:
                pred = detections_pred.iloc[i]
                row_idx = np.min(np.nonzero(row_idxs == i)[0])  # row_idxs.index(i)
                gt = detections_gt.iloc[col_idxs[row_idx]]
                metric = cost_matrix[i, col_idxs[row_idx]]
            self.draw_detection(image, pred, gt, metric)


    @abstractmethod
    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        pass

    def color(self, detection, is_prediction, color_type="bbox"):
        assert self.colors is not None
        if color_type not in self.colors:
            raise ValueError(f"{color_type} not declared in the colors dict for visualization")
        cmap = prediction_cmap if is_prediction else ground_truth_cmap
        if pd.isna(detection.track_id):
            color = self.colors[color_type].no_id
        else:
            cmap_key = "prediction" if is_prediction else "ground_truth"
            color_id = cmap[int(detection.track_id) % len(cmap)]
            color = self.colors[color_type][cmap_key] or color_id

        return color
