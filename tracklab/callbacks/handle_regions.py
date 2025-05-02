import cv2
import numpy as np
import pandas as pd

from tracklab.callbacks import Callback


class IgnoredRegions(Callback):
    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        image_metadatas = engine.img_metadatas[
            engine.img_metadatas.video_id == video_idx
        ]
        """
        detections.insert(-1, "ignored", detections.apply(
            lambda x: self.mark_ignored(x, image_metadatas), axis=1
        ))
        """
        if len(detections):
            detections["ignored"] = detections.apply(
                lambda x: self.mark_ignored(x, image_metadatas), axis=1
            )
        else:
            detections["ignored"] = pd.NA

    def __init__(self, max_intersection=0.9):
        self.max_intersection = max_intersection

    def mark_ignored(self, detection, image_metadatas):
        if hasattr(image_metadatas, "ignore_regions_x") and hasattr(
            image_metadatas, "ignore_regions_y"
        ):
            image_metadata = image_metadatas.loc[detection.image_id]
            return self.compute_iou(
                detection.bbox.ltrb(rounded=True),
                image_metadata.ignore_regions_x,
                image_metadata.ignore_regions_y,
            )
        return False

    def compute_iou(self, bbox_ltrb, ignore_regions_x, ignore_regions_y):
        """Compute the intersection of a detection and a list of ignore regions and check whether
        it is higher than a portion of the area or not.

        Args:
            bbox_ltrb (np.array): bounding box of the detection [left, top, right, bottom]
            ignore_regions_x (tuple): list of ignore regions x coordinates
            ignore_regions_y (tuple): list of ignore regions y coordinates

        Returns:
            bool: True if the area of the detection is higher than a certain threshold in an ignore region,
            False otherwise
        """
        l, t, r, b = bbox_ltrb

        for ignore_region_x, ignore_region_y in zip(ignore_regions_x, ignore_regions_y):
            polygon_points = (
                np.array([ignore_region_x, ignore_region_y]).round().T.astype(int)
            )
            image_dim_max = (
                max(b, polygon_points[:, 1].max()),
                max(r, polygon_points[:, 0].max()),
            )  # height, width
            bbox_mask = np.zeros(image_dim_max, dtype=np.uint8)
            bbox_mask[t:b, l:r] = 1
            ignore_mask = np.zeros(image_dim_max, dtype=np.uint8)
            ignore_mask = cv2.fillPoly(ignore_mask, [polygon_points], 1)
            intersection_area = np.logical_and(bbox_mask, ignore_mask).sum()
            bbox_area = (r - l) * (b - t)
            if intersection_area > self.max_intersection * bbox_area:
                return True
        return False
