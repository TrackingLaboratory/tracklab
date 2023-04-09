import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pbtrack import (
    TrackingDataset,
    TrackingSet,
    ImageMetadatas,
    VideoMetadatas,
    Detections,
)


class PoseTrack18(TrackingDataset):
    def __init__(self, dataset_path: str, annotation_path: str, *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), "'{}' directory does not exist".format(
            self.dataset_path
        )
        self.annotation_path = Path(annotation_path)
        assert self.annotation_path.exists(), "'{}' directory does not exist".format(
            self.annotation_path
        )

        train_set = load_tracking_set(self.annotation_path, self.dataset_path, "train")
        val_set = load_tracking_set(self.annotation_path, self.dataset_path, "val")
        test_set = None  # TODO load

        super().__init__(dataset_path, train_set, val_set, test_set, *args, **kwargs)


def load_tracking_set(anns_path, dataset_path, split):
    # Load annotations into Pandas dataframes
    video_metadatas, image_metadatas, detections = load_annotations(anns_path, split)
    # Fix formatting of dataframes to be compatible with pbtrack
    video_metadatas, image_metadatas, detections = fix_formatting(
        video_metadatas, image_metadatas, detections, dataset_path
    )
    return TrackingSet(
        split,
        VideoMetadatas(video_metadatas),
        ImageMetadatas(image_metadatas),
        Detections(detections),
    )


def load_annotations(anns_path, split):
    anns_path = anns_path / split
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, "No annotations files found in {}".format(
        anns_path
    )
    detections = []
    image_metadatas = []
    video_metadatas = []
    for path in anns_files_list:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            detections.extend(data_dict["annotations"])
            image_metadatas.extend(data_dict["images"])
            categories = data_dict["categories"]
            video_metadata = {
                "id": data_dict["images"][0]["vid_id"],
                "name": path.stem,
                "categories": categories,
            }
            video_metadatas.append(video_metadata)

    return (
        pd.DataFrame(video_metadatas),
        pd.DataFrame(image_metadatas),
        pd.DataFrame(detections),
    )


def fix_formatting(video_metadatas, image_metadatas, detections, dataset_path):
    # Videos
    video_metadatas.set_index("id", drop=False, inplace=True)

    # Images
    image_metadatas.drop(["frame_id"], axis=1, inplace=True)  # id == image_id
    image_metadatas["video_name"] = image_metadatas["file_name"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    image_metadatas["file_name"] = image_metadatas["file_name"].apply(
        lambda x: os.path.join(dataset_path, x)  # FIXME use relative path
    )
    image_metadatas["frame"] = image_metadatas["file_name"].apply(
        lambda x: int(os.path.basename(x).split(".")[0]) + 1
    )
    image_metadatas.rename(
        columns={"vid_id": "video_id", "file_name": "file_path", "nframes": "nframes"},
        inplace=True,
    )
    image_metadatas.set_index("id", drop=False, inplace=True)

    # Detections
    detections.drop(["bbox_head"], axis=1, inplace=True)
    detections.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
    detections.bbox_ltwh = detections.bbox_ltwh.apply(lambda x: np.array(x))
    detections.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
    detections.keypoints_xyc = detections.keypoints_xyc.apply(
        lambda x: np.reshape(np.array(x), (-1, 3))
    )
    detections.set_index("id", drop=False, inplace=True)
    # compute detection visiblity as average keypoints visibility
    detections["visibility"] = detections.keypoints_xyc.apply(lambda x: x[:, 2].mean())
    # add video_id to detections, will be used for bpbreid 'camid' parameter
    detections = detections.merge(
        image_metadatas[["video_id"]], how="left", left_on="image_id", right_index=True
    )

    return video_metadatas, image_metadatas, detections
