import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pbtrack.datastruct import TrackingDataset, TrackingSet


class MOT20(TrackingDataset):
    def __init__(self, dataset_path: str, *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), "'{}' directory does not exist".format(
            self.dataset_path
        )

        train_set = load_train(self.dataset_path / "train")
        val_set = None  # TODO
        test_set = load_test(self.dataset_path / "test")

        super().__init__(dataset_path, train_set, val_set, test_set, *args, **kwargs)


def load_train(dataset_path):
    videos = [x for x in dataset_path.iterdir() if x.is_dir()]

    video_metadatas = []
    image_metadatas = []
    detections = []
    for video_id, video in enumerate(videos):
        image_id_offset = video_id * 100000

        # image metadata
        all_images = list(video.glob("img1/*.jpg"))
        for frame, image in enumerate(all_images):
            frame += 1
            image_metadatas.append(
                {
                    "id": image_id_offset + frame,
                    "video_id": video_id,
                    "file_path": str(image),
                    "frame": frame,
                }
            )

        # detections
        gt_path = video / "gt" / "gt.txt"
        with open(gt_path) as f:
            gt = f.readlines()
            gt = [x.strip().split(",") for x in gt]
            gt = np.array(gt).astype(float)

            for detection in gt:
                detections.append(
                    {
                        "image_id": int(image_id_offset + detection[0]),
                        "video_id": video_id,
                        "track_id": int(detection[1]),
                        "bbox_ltwh": detection[2:6],
                        "category_id": 1,
                    }
                )

        # video metadata
        video_metadatas.append(
            {
                "id": video_id,
                "name": video.name,
                "categories": [{"supercategory": "person", "id": 1, "name": "person"}],
                "nframes": len(all_images),
            }
        )

    return TrackingSet(
        "train",
        pd.DataFrame(video_metadatas).set_index("id"),
        pd.DataFrame(image_metadatas).set_index("id"),
        pd.DataFrame(detections),
    )


def load_test(dataset_path):
    videos = [x for x in dataset_path.iterdir() if x.is_dir()]

    video_metadatas = []
    image_metadatas = []
    detections = []
    for video_id, video in enumerate(videos):
        video_id += 10
        image_id_offset = video_id * 100000

        # image metadata
        all_images = list(video.glob("img1/*.jpg"))
        for frame, image in enumerate(all_images):
            frame += 1
            image_metadatas.append(
                {
                    "id": image_id_offset + frame,
                    "video_id": video_id,
                    "file_path": str(image),
                    "frame": frame,
                }
            )

        # detections
        gt_path = video / "det" / "det.txt"
        with open(gt_path) as f:
            gt = f.readlines()
        gt = [x.strip().split(",") for x in gt]
        gt = np.array(gt).astype(float)
        for detection in gt:
            detections.append(
                {
                    "image_id": int(image_id_offset + detection[0]),
                    "video_id": video_id,
                    "track_id": np.nan,
                    "bbox_ltwh": detection[2:6],
                    "category_id": 1,
                }
            )

        # video metadata
        video_metadatas.append(
            {
                "id": video_id,
                "name": video.name,
                "categories": [{"name": "person", "id": 1}],
                "nframes": len(all_images),
            }
        )

    return TrackingSet(
        "test",
        pd.DataFrame(video_metadatas).set_index("id"),
        pd.DataFrame(image_metadatas).set_index("id"),
        pd.DataFrame(detections),
    )
