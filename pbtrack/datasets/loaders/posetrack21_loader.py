from __future__ import absolute_import, division, print_function

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pbtrack.datasets.loaders import DatasetLoader
from pbtrack.tracker.categories import Categories
from pbtrack.tracker.detections import Detections
from pbtrack.tracker.images import Images
from pbtrack.utils.coordinates import (
    kp_img_to_kp_bbox,
)
from hydra.utils import to_absolute_path

sys.path.append(to_absolute_path("modules/reid/bpbreid"))


def load_annotations(anns_path, splits):
    detections_list = []
    categories_list = []
    images_list = []
    for split in splits:
        anns_split_path = anns_path / split
        anns_files_list = list(anns_split_path.glob("*.json"))
        assert len(anns_files_list) > 0, "No annotations files found in {}".format(
            anns_split_path
        )
        for path in anns_files_list:
            json_file = open(path)
            data_dict = json.load(json_file)
            detections = pd.DataFrame(data_dict["annotations"])
            categories = pd.DataFrame(data_dict["categories"])
            images = pd.DataFrame(data_dict["images"])
            detections['split'] = split
            categories['split'] = split
            images['split'] = split
            detections_list.append(detections)
            categories_list.append(categories)
            images_list.append(images)
    detections = pd.concat(detections_list).reset_index(drop=True)
    categories = pd.concat(categories_list).reset_index(drop=True)
    images = pd.concat(images_list).reset_index(drop=True)
    return categories, detections, images


def fix_formatting(images, categories, detections):
    detections.bbox = detections.bbox.apply(lambda x: np.array(x))
    detections.bbox_head = detections.bbox_head.apply(lambda x: np.array(x))
    # reshape keypoints to (n, 3) array
    detections.keypoints = detections.keypoints.apply(
        lambda kp: np.array(kp).reshape(-1, 3)
    )
    # If detections do not have a unique 'id' column, add one for further unambiguous detection referencing
    if "id" not in detections:
        detections["id"] = detections.index
    # compute detection visiblity as average keypoints visibility
    detections["visibility"] = detections.keypoints.apply(lambda x: x[:, 2].mean())
    # precompute various bbox formats
    detections.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
    detections["bbox_ltrb"] = detections.bbox_ltwh.apply(
        lambda ltwh: np.concatenate((ltwh[:2], ltwh[:2] + ltwh[2:]))
    )
    detections["bbox_cxcywh"] = detections.bbox_ltwh.apply(
        lambda ltwh: np.concatenate((ltwh[:2] + ltwh[2:] / 2, ltwh[2:]))
    )
    # precompute various keypoints formats
    detections.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
    detections["keypoints_bbox_xyc"] = detections.apply(
        lambda r: kp_img_to_kp_bbox(r.keypoints_xyc, r.bbox_ltwh), axis=1
    )

    # rename base 'vid_id' to 'video_id', to be consistent with 'image_id'
    images.rename(columns={"vid_id": "video_id"}, inplace=True)
    # add video_id to gt_dets, will be used for torchreid 'camid' paremeter
    detections = detections.merge(
        images[["image_id", "video_id"]], on="image_id", how="left"
    )

    return images, categories, detections


class PoseTrack21Loader(DatasetLoader):
    splits = {
        'train': {"build_reid_meta": True, "is_test_set": False},
        'val': {"build_reid_meta": True, "is_test_set": True},
        # 'test': {"build_reid_meta": False, "is_test_set": True},
    }
    annotations_dir = Path('posetrack_data')

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.anns_path = self.dataset_path / self.annotations_dir
        assert self.anns_path.exists(), "Annotations path does not exist in '{}'".format(self.anns_path)

    def load_dataset(self):
        # Load annotations into Pandas dataframes
        categories, detections, images = load_annotations(self.anns_path, self.splits)
        # Fix formatting of dataframes to be compatible with pbtrack
        images, categories, detections = fix_formatting(images, categories, detections)
        return Images(images), Categories(categories), Detections(detections)
