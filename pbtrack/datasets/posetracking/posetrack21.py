from __future__ import absolute_import, division, print_function

import sys
import json
import numpy as np
from pathlib import Path

from pbtrack.datasets.tracking_dataset import TrackingDataset, TrackingSet
from pbtrack.datastruct.images import Image, Images
from pbtrack.datastruct.detections import Detection, Detections
from pbtrack.datastruct.categories import Categorie, Categories

from hydra.utils import to_absolute_path
sys.path.append(to_absolute_path("plugins/reid/bpbreid"))

class PoseTrack21(TrackingDataset):
    annotations_dir = 'posetrack_data'
    def __init__(self, dataset_path: str):
        self.anns_path = Path(dataset_path) / self.annotations_dir
        assert self.anns_path.exists(), "Annotations path does not exist in '{}'".format(self.anns_path)
        train_set = load_set(self.anns_path, dataset_path, "train")
        val_set = load_set(self.anns_path, dataset_path, "val")
        test_set = None  # TODO no json, load images
        super().__init__(dataset_path, train_set, val_set, test_set)  # type: ignore

def load_set(anns_path, dataset_path, split):
    # Load annotations into Pandas dataframes
    images, detections, categories = load_annotations(anns_path, dataset_path, split)
    return TrackingSet(split, Images(images), Detections(detections), Categories(categories))

def load_annotations(anns_path, dataset_path, split):
    anns_path = anns_path / split
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, "No annotations files found in {}".format(
        anns_path
    )
    images_list = []
    detections_list = []
    categories_list = []
    for path in anns_files_list:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            for frame, image in enumerate(data_dict['images']):
                images_list.append(
                    Image(
                        id = image['image_id'],
                        video_id = image['vid_id'],
                        frame = frame,
                        nframe = image['nframes'],
                        file_path = Path(dataset_path) / image['file_name'],
                        is_labeled = image['is_labeled'],
                        ignore_regions_x = image['ignore_regions_x'],
                        ignore_regions_y = image['ignore_regions_y']
                    )  # type: ignore
                )
            for annotation in data_dict['annotations']:
                detections_list.append(
                    Detection(
                        image_id = annotation['image_id'],
                        id = annotation['id'],
                        bbox = np.array(annotation['bbox']),
                        keypoints_xyc = np.reshape(np.array(annotation['keypoints']), (-1, 3)),
                        person_id = annotation['person_id'],
                        track_id = annotation['track_id'],
                        category_id = annotation['category_id'],
                    )  # type: ignore
                )
            for category in data_dict['categories']:
                categories_list.append(
                    Categorie(
                        id = category['id'],
                        name = category['name'],
                        supercategory = category['supercategory'],
                        keypoints = category['keypoints'],
                        skeleton = category['skeleton'],
                    )  # type: ignore
                )
    return images_list, detections_list, categories_list
