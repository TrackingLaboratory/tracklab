import os
import sys
import json
import numpy as np
import pandas as pd

from pathlib import Path

from pbtrack.core.evaluator import Evaluator
from pbtrack.datastruct.tracking_dataset import TrackingDataset, TrackingSet
from pbtrack.datastruct.image_metadatas import ImageMetadatas
from pbtrack.datastruct.video_metadatas import VideoMetadatas
from pbtrack.datastruct.detections import Detections


from hydra.utils import to_absolute_path

sys.path.append(to_absolute_path("plugins/eval/PoseTrack21/eval/posetrack21"))
from posetrack21.trackeval import (
    PoseEvaluator,
    Evaluator,
    EvaluatorMOT,
    EvaluatorReid,
)
from posetrack21.trackeval.datasets import PoseTrack, PoseTrackMOT
from posetrack21.trackeval.metrics import PosemAP, HOTA, HOTAeypoints, HOTAReidKeypoints


class PoseTrack21(TrackingDataset):
    annotations_dir = "posetrack_data"

    def __init__(self, dataset_path: str, *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), "Dataset path does not exist in '{}'".format(
            self.dataset_path
        )
        self.anns_path = self.dataset_path / self.annotations_dir
        assert (
            self.anns_path.exists()
        ), "Annotations path does not exist in '{}'".format(self.anns_path)

        train_set = load_tracking_set(self.anns_path, self.dataset_path, "train")
        val_set = load_tracking_set(self.anns_path, self.dataset_path, "val")
        test_set = None  # TODO no json, load images

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
    image_metadatas.drop(["image_id"], axis=1, inplace=True)  # id == image_id
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
        columns={"vid_id": "video_id", "file_name": "file_path", "nframes": "nframe"},
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


class PoseTrack21Eval(Evaluator):
    def __init__(self, cfg, tracking_dataset):
        self.cfg = cfg
        self.tracking_dataset = tracking_dataset

        # populate evaluators & metrics
        if cfg.eval_pose_estimation:
            self.pose_estimation_evaluator = PoseEvaluator(cfg.posetrack_evaluator)
            self.pose_estimation_metrics = [PosemAP()]
        if cfg.eval_mot:
            self.mot_evaluator = EvaluatorMOT(cfg.mot_evaluator)
            self.mot_metrics = [HOTA()]
        if cfg.eval_pose_tracking:
            self.pose_tracking_evaluator = Evaluator(cfg.posetrack_evaluator)
            self.pose_tracking_metrics = [HOTAeypoints()]
        if cfg.eval_reid_pose_tracking:
            self.reid_pose_tracking_evaluator = EvaluatorReid(cfg.posetrack_evaluator)
            self.reid_pose_tracking_metrics = [HOTAReidKeypoints()]

    def run(self):
        if self.cfg.eval_mot:  # MOT
            mot_tracker_path = self.cfg.mot_dataset.TRACKERS_FOLDER
            mot_gt_path = self.cfg.mot_dataset.GT_FOLDER
            mot_datasets = []
            if self.cfg.on_val:  # val
                # FIXME change from tracking_dataset to tracker_state
                val_df = pd.merge(
                    self.tracking_dataset.val_set.detections.reset_index(drop=True),
                    self.tracking_dataset.val_set.image_metadatas.reset_index(
                        drop=True
                    ),
                    left_on="image_id",
                    right_on="id",
                )
                mot_val_tracker_path = os.path.join(mot_tracker_path, "val", "mot")
                os.makedirs(mot_val_tracker_path, exist_ok=True)
                self._save_mot(val_df, mot_val_tracker_path)

                self.cfg.mot_dataset.GT_FOLDER = os.path.join(mot_gt_path, "val")
                self.cfg.mot_dataset.TRACKERS_FOLDER = os.path.join(
                    mot_tracker_path, "val"
                )
                mot_datasets.append(PoseTrackMOT(self.cfg.mot_dataset))

            if self.cfg.on_train:  # train
                # FIXME change from tracking_dataset to tracker_state
                train_df = pd.merge(
                    self.tracking_dataset.train_set.detections.reset_index(drop=True),
                    self.tracking_dataset.train_set.image_metadatas.reset_index(
                        drop=True
                    ),
                    left_on="image_id",
                    right_on="id",
                )
                mot_train_tracker_path = os.path.join(mot_tracker_path, "train", "mot")
                os.makedirs(mot_train_tracker_path, exist_ok=True)
                self._save_mot(train_df, mot_train_tracker_path)

                self.cfg.mot_dataset.GT_FOLDER = os.path.join(mot_gt_path, "train")
                self.cfg.mot_dataset.TRACKERS_FOLDER = os.path.join(
                    mot_tracker_path, "train"
                )
                mot_datasets.append(PoseTrackMOT(self.cfg.mot_dataset))

            self.mot_evaluator.evaluate(mot_datasets, self.mot_metrics)

        if any(
            (
                self.cfg.eval_pose_estimation,
                self.cfg.eval_pose_tracking,
                self.cfg.eval_reid_pose_tracking,
            )
        ):  # PoseTrack

            posetrack_tracker_path = self.cfg.posetrack_dataset.TRACKERS_FOLDER
            posetrack_gt_path = self.cfg.posetrack_dataset.GT_FOLDER
            posetrack_datasets = []

            if self.cfg.on_val:  # val
                posetrack_val_tracker_path = os.path.join(posetrack_tracker_path, "val")
                os.makedirs(posetrack_val_tracker_path, exist_ok=True)
                # FIXME change from tracking_dataset to tracker_state
                self._save_posetrack(
                    self.tracking_dataset.val_set.detections,
                    self.tracking_dataset.val_set.image_metadatas,
                    posetrack_val_tracker_path,
                )

                self.cfg.posetrack_dataset.GT_FOLDER = os.path.join(
                    posetrack_gt_path, "val"
                )
                self.cfg.posetrack_dataset.TRACKERS_FOLDER = posetrack_val_tracker_path
                posetrack_datasets.append(PoseTrack(self.cfg.posetrack_dataset))

            if self.cfg.on_train:  # train
                posetrack_train_tracker_path = os.path.join(
                    posetrack_tracker_path, "train"
                )
                os.makedirs(posetrack_train_tracker_path, exist_ok=True)
                # FIXME change from tracking_dataset to tracker_state
                self._save_posetrack(
                    self.tracking_dataset.train_set.detections,
                    self.tracking_dataset.train_set.image_metadatas,
                    posetrack_train_tracker_path,
                )

                self.cfg.posetrack_dataset.GT_FOLDER = os.path.join(
                    posetrack_gt_path, "train"
                )
                self.cfg.posetrack_dataset.TRACKERS_FOLDER = (
                    posetrack_train_tracker_path
                )
                posetrack_datasets.append(PoseTrack(self.cfg.posetrack_dataset))

            if self.cfg.eval_pose_estimation:
                self.pose_estimation_evaluator.evaluate(
                    posetrack_datasets, self.pose_estimation_metrics
                )
            if self.cfg.eval_pose_estimation:
                self.pose_tracking_evaluator.evaluate(
                    posetrack_datasets, self.pose_tracking_metrics
                )
            if self.cfg.eval_reid_pose_tracking:
                self.reid_pose_tracking_evaluator.evaluate(
                    posetrack_datasets, self.reid_pose_tracking_metrics
                )

    def _save_mot(self, df, save_path):
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        videos_names = df["video_name"].unique()
        for video_name in videos_names:
            file_path = os.path.join(save_path, f"{video_name}.txt")
            video_df = df[df["video_name"] == video_name]
            video_df.sort_values(by="frame", inplace=True)
            video_df["bb_left"] = video_df["bbox_ltwh"].apply(lambda x: x[0])
            video_df["bb_top"] = video_df["bbox_ltwh"].apply(lambda x: x[1])
            video_df["bb_width"] = video_df["bbox_ltwh"].apply(lambda x: x[2])
            video_df["bb_height"] = video_df["bbox_ltwh"].apply(lambda x: x[0])
            video_df["conf"] = video_df["keypoints_xyc"].apply(
                lambda x: np.mean(x[:, 2])
            )
            video_df = video_df.assign(x=-1, y=-1, z=-1)
            video_df.to_csv(
                file_path,
                columns=[
                    "frame",
                    "track_id",
                    "bb_left",
                    "bb_top",
                    "bb_width",
                    "bb_height",
                    "conf",
                    "x",
                    "y",
                    "z",
                ],
                header=False,
                index=False,
            )

    def _save_posetrack(self, detections, image_metadatas, save_path):
        """{
            "images": [
                {
                    "file_name": "images/train/000001_bonn_train/000000.jpg",
                    "id": 10000010000,
                    "frame_id": 10000010000
                },
            ],
            "annotations": [
                {
                    "bbox": [x1, y1, w, h],
                    "image_id": 10000010000,
                    "keypoints": [x1, y1, vis1, ..., x17, y17, vis17],
                    "scores": [s1, ..., s17],
                    "person_id": 1024,
                    "track_id": 0
                }
            ]
        }"""

        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            file_path = os.path.join(save_path, f"{video_name}.json")
            # record images
            image_metadatas_video = image_metadatas[
                image_metadatas["video_name"] == video_name
            ]
            image_metadatas_video.rename(
                columns={"file_path": "file_name"}, inplace=True
            )
            image_metadatas_video["frame_id"] = image_metadatas_video["id"]
            images = image_metadatas_video[["file_name", "id", "frame_id"]].to_dict(
                "record"
            )
            # record annotations
            detections_video = detections[
                detections["image_id"].isin(image_metadatas_video["id"])
            ]
            detections_video.rename(
                columns={"keypoints_xyc": "keypoints", "bbox_ltwh": "bbox"},
                inplace=True,
            )
            detections_video["scores"] = detections_video["keypoints"].apply(
                lambda x: x[:, 2]
            )
            annotations = detections_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("record")
            data = {"images": images, "annotations": annotations}
            with open(file_path, "w") as f:
                json.dump(data, f, cls=PoseTrackEncoder)


class PoseTrackEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return json.JSONEncoder.default(self, obj)
