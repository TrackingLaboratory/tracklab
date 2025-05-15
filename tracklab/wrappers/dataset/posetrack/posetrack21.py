import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from tracklab.datastruct import TrackingDataset, TrackingSet


class PoseTrack21(TrackingDataset):
    """
    Train set: 43603 images
    Val set: 20161 images
    Test set: ??? images
    """

    name = "posetrack21"
    nickname = "ptt"

    def __init__(
        self,
        dataset_path: str,
        annotation_path: str,
        posetrack_version=21,
        *args,
        **kwargs
    ):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), "'{}' directory does not exist".format(
            self.dataset_path
        )
        self.annotation_path = Path(annotation_path)
        assert self.annotation_path.exists(), "'{}' directory does not exist".format(
            self.annotation_path
        )

        train_set = load_tracking_set(
            self.annotation_path / "train", self.dataset_path, posetrack_version
        )
        val_set = load_tracking_set(
            self.annotation_path / "val", self.dataset_path, posetrack_version
        )
        test_set = None  # TODO

        sets = {"train": train_set, "val": val_set, "test": test_set}

        super().__init__(dataset_path, sets, *args, **kwargs)


def load_tracking_set(anns_path, dataset_path, posetrack_version=21):
    # Load annotations into Pandas dataframes
    video_metadatas, image_metadatas, detections_gt = load_annotations(anns_path)
    # Fix formatting of dataframes to be compatible with tracklab
    video_metadatas, image_metadatas, detections_gt = fix_formatting(
        video_metadatas, image_metadatas, detections_gt, dataset_path, posetrack_version
    )
    return TrackingSet(
        video_metadatas,
        image_metadatas,
        detections_gt,
    )


def load_annotations(anns_path):
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, "No annotations files found in {}".format(
        anns_path
    )
    detections_gt = []
    image_metadatas = []
    video_metadatas = []
    for path in anns_files_list:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            detections_gt.extend(data_dict["annotations"])
            image_metadatas.extend(data_dict["images"])
            video_metadatas.append(
                {
                    "id": data_dict["images"][0]["vid_id"],
                    "nframes": len(data_dict["images"]),
                    "name": path.stem,
                    "categories": data_dict["categories"],
                }
            )

    return (
        pd.DataFrame(video_metadatas),
        pd.DataFrame(image_metadatas),
        pd.DataFrame(detections_gt),
    )


def fix_formatting(
    video_metadatas, image_metadatas, detections_gt, dataset_path, posetrack_version
):
    image_id = "image_id" if posetrack_version == 21 else "frame_id"

    # Videos
    video_metadatas.set_index("id", drop=True, inplace=True)

    # Images
    image_metadatas["file_name"] = image_metadatas["file_name"].apply(
        lambda x: os.path.join(dataset_path, x)
    )
    image_metadatas["frame"] = image_metadatas["file_name"].apply(
        lambda x: int(os.path.basename(x).split(".")[0]) + 1
    )
    image_metadatas.rename(
        columns={"vid_id": "video_id", "file_name": "file_path"},
        inplace=True,
    )
    image_metadatas.set_index("id", drop=True, inplace=True)

    # Detections
    detections_gt.drop(["bbox_head"], axis=1, inplace=True)
    detections_gt.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
    detections_gt.bbox_ltwh = detections_gt.bbox_ltwh.apply(lambda x: np.array(x))
    detections_gt.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
    detections_gt.keypoints_xyc = detections_gt.keypoints_xyc.apply(
        lambda x: np.reshape(np.array(x), (-1, 3))
    )
    detections_gt.set_index("id", drop=True, inplace=True)
    detections_gt = detections_gt.merge(
        image_metadatas[["video_id"]], how="left", left_on="image_id", right_index=True
    )

    return video_metadatas, image_metadatas, detections_gt
