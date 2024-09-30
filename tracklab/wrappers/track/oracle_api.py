import logging
import pandas as pd
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tracklab.pipeline import ImageLevelModule

log = logging.getLogger(__name__)

INFTY_COST = 1e+5

class Oracle(ImageLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = [
        "track_id",
    ]

    def __init__(self,
                 cfg,
                 device,
                 tracking_dataset,
                 **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.tracking_dataset = tracking_dataset

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        return []

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return []

        image_id = metadatas.id.unique()[0]
        video_id = metadatas.video_id.unique()[0]

        # Get ground truth detections for the current image
        all_detections_gt = self.tracking_dataset.sets[self.cfg['eval_set']].detections_gt
        detections_gt = all_detections_gt[all_detections_gt.image_id == image_id]

        # Ensure the video id matches
        assert detections_gt.video_id.unique()[0] == video_id

        if self.cfg['return_gt']:
            # Increment index of detections_gt by all_detections_gt max index
            detections_gt.index += all_detections_gt.index.max() + 1
            return detections_gt

        # If no ground truth detections, return empty
        if len(detections_gt) == 0:
            return []

        # Extract bounding boxes from ground truth and detected boxes
        bbox_ltwh_gt = np.vstack(detections_gt.bbox_ltwh.values)
        bbox_ltwh = np.vstack(detections.bbox_ltwh.values)

        assert len(bbox_ltwh_gt) > 0 and len(bbox_ltwh) > 0

        # Compute IoU cost matrix (cost is 1 - IoU)
        cost_matrix = 1 - compute_iou_matrix(bbox_ltwh_gt, bbox_ltwh)

        # Apply threshold to the cost matrix: costs above 0.5 are set to 1
        cost_matrix[cost_matrix > 0.5] = INFTY_COST

        # Perform the Hungarian algorithm (linear sum assignment) to get matches
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # After assignment, filter out any assignments with a cost of 1 (invalid matches)
        valid_matches = cost_matrix[row_ind, col_ind] < 1.0
        row_ind = row_ind[valid_matches]
        col_ind = col_ind[valid_matches]

        # Retrieve matched ground truth and detection rows
        matched_gt = detections_gt.iloc[row_ind]
        matched_detections = detections.iloc[col_ind]

        # Assign track IDs from ground truth to detections
        matched_detections['track_id'] = matched_gt['track_id'].values

        return matched_detections


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute the IoU matrix between two arrays of bounding boxes in the format "ltwh".

    Args:
        boxes1 (np.ndarray): Array of bounding boxes with shape [N, 4], each row is [left, top, width, height].
        boxes2 (np.ndarray): Array of bounding boxes with shape [M, 4], each row is [left, top, width, height].

    Returns:
        iou_matrix (np.ndarray): IoU matrix of shape [N, M] where each element [i, j] is the IoU between boxes1[i] and boxes2[j].
    """

    # Convert boxes from "ltwh" (left, top, width, height) to "ltrb" (left, top, right, bottom)
    boxes1_ltrb = np.concatenate([boxes1[:, :2], boxes1[:, :2] + boxes1[:, 2:4]], axis=1)  # [N, 4]
    boxes2_ltrb = np.concatenate([boxes2[:, :2], boxes2[:, :2] + boxes2[:, 2:4]], axis=1)  # [M, 4]

    # Expand dimensions for broadcasting
    boxes1_ltrb = np.expand_dims(boxes1_ltrb, axis=1)  # [N, 1, 4]
    boxes2_ltrb = np.expand_dims(boxes2_ltrb, axis=0)  # [1, M, 4]

    # Compute intersection coordinates
    left_top = np.maximum(boxes1_ltrb[..., :2], boxes2_ltrb[..., :2])  # [N, M, 2]
    right_bottom = np.minimum(boxes1_ltrb[..., 2:], boxes2_ltrb[..., 2:])  # [N, M, 2]

    # Compute intersection area
    intersection_dims = np.clip(right_bottom - left_top, a_min=0, a_max=None)  # [N, M, 2]
    intersection_area = intersection_dims[..., 0] * intersection_dims[..., 1]  # [N, M]

    # Compute areas of the individual boxes
    area_boxes1 = (boxes1[:, 2] * boxes1[:, 3]).reshape(-1, 1)  # [N, 1]
    area_boxes2 = (boxes2[:, 2] * boxes2[:, 3]).reshape(1, -1)  # [1, M]

    # Compute union area
    union_area = area_boxes1 + area_boxes2 - intersection_area  # [N, M]

    # Compute IoU
    iou_matrix = intersection_area / union_area  # [N, M]

    return iou_matrix