import os
import sys
import json
import numpy as np
import pandas as pd

from pathlib import Path
from pbtrack.datastruct.tracking_dataset import TrackingDataset, TrackingSet
from pbtrack.datastruct.image_metadatas import ImageMetadatas
from pbtrack.datastruct.video_metadatas import VideoMetadatas
from pbtrack.datastruct.detections import Detections
from hydra.utils import to_absolute_path

sys.path.append(to_absolute_path("plugins/reid/bpbreid"))


class PoseTrack21(TrackingDataset):
    annotations_dir = "posetrack_data"

    def __init__(self, dataset_path: str):
        self.anns_path = Path(dataset_path) / self.annotations_dir
        assert (
            self.anns_path.exists()
        ), "Annotations path does not exist in '{}'".format(self.anns_path)
        train_set = load_set(self.anns_path, dataset_path, "train")
        val_set = load_set(self.anns_path, dataset_path, "val")
        test_set = None  # TODO no json, load images
        super().__init__(dataset_path, train_set, val_set, test_set)  # type: ignore


def load_set(anns_path, dataset_path, split):
    # Load annotations into Pandas dataframes
    detections, image_metadatas, video_metadatas = load_annotations(anns_path, dataset_path, split)
    return TrackingSet(split, detections, image_metadatas, video_metadatas)

def load_annotations(anns_path, dataset_path, split):
    anns_path = anns_path / split
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, "No annotations files found in {}".format(
        anns_path
    )
    detections = []
    image_metadatas = []
    video_metadatas = []
    for path in anns_files_list[:]:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            detections.extend(data_dict["annotations"])
            image_metadatas.extend(data_dict["images"])
            video_metadatas.extend(data_dict["categories"])
    # Detections
    detections = pd.DataFrame(detections)
    detections = detections.drop(["bbox_head"], axis=1)
    detections["keypoints"] = detections["keypoints"].apply(
        lambda x: np.reshape(np.array(x), (-1, 3))
    )
    detections = detections.rename(columns={"keypoints": "keypoints_xyc"})
    detections = Detections(detections)
    # Images
    image_metadatas = pd.DataFrame(image_metadatas)
    image_metadatas = image_metadatas.drop(["image_id", "has_labeled_person"], axis=1)
    image_metadatas["file_name"] = image_metadatas["file_name"].apply(
        lambda x: os.path.join(dataset_path, x)
    )
    image_metadatas["frame"] = image_metadatas["file_name"].apply(
        lambda x: int(os.path.basename(x).split(".")[0]) + 1
    )
    image_metadatas = image_metadatas.rename(
        columns={"vid_id": "video_id", "file_name": "file_path", "nframes": "nframe"}
    )
    image_metadatas.set_index("id", drop=False, inplace=True)
    image_metadatas = ImageMetadatas(image_metadatas)
    # Videos
    video_metadatas = pd.DataFrame(video_metadatas)
    video_metadatas = VideoMetadatas(video_metadatas)
    return TrackingSet(split, detections, image_metadatas, video_metadatas)
