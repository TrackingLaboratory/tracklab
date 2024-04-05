from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou


class Visualizer(ABC):
    @abstractmethod
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        pass


class ImageVisualizer(Visualizer, ABC):
    pass


class DetectionVisualizer(Visualizer, ABC):
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


