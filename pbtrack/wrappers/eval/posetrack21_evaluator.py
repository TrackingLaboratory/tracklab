import os
import sys
import json
import numpy as np
import pandas as pd

from pbtrack.core.evaluator import Evaluator as EvaluatorBase

import pbtrack
from pathlib import Path

root_dir = Path(pbtrack.__file__).parents[1]
sys.path.append(
    str((root_dir / "plugins/eval/PoseTrack21/eval/mot").resolve())
)  # FIXME : ugly
from datasets.pt_warper import PTWrapper
from evaluate_mot import get_mot_accum, evaluate_mot_accums

sys.path.append(
    str((root_dir / "plugins/eval/PoseTrack21/eval/posetrack21").resolve())
)  # FIXME : ugly
from posetrack21.trackeval import (
    PoseEvaluator,
    Evaluator,
    EvaluatorMOT,
    EvaluatorReid,
)
from posetrack21.trackeval.datasets import PoseTrack, PoseTrackMOT
from posetrack21.trackeval.metrics import (
    PosemAP,
    HOTA,
    HOTAeypoints,
    HOTAReidKeypoints,
)


class PoseTrack21(EvaluatorBase):
    def __init__(self, cfg):
        self.cfg = cfg

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

    def run(self, tracker_state):
        if self.cfg.eval_mot:
            # populate dataset
            mot_tracker_path = os.path.join(self.cfg.mot_dataset.TRACKERS_FOLDER, "mot")
            os.makedirs(mot_tracker_path, exist_ok=True)
            mot_df = self._mot_encoding(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            self._save_mot(mot_df, mot_tracker_path)
            self.cfg.mot_dataset.TRACKERS_FOLDER = os.path.abspath(self.cfg.mot_dataset.TRACKERS_FOLDER)
            # run evaluator - HOTA
            self.mot_evaluator.evaluate(
                [PoseTrackMOT(self.cfg.mot_dataset)], self.mot_metrics
            )
            # run evaluator - MOTA
            dataset = PTWrapper(
                self.cfg.mot_dataset.GT_FOLDER,
                self.cfg.mot_evaluator.dataset_path,
                self.cfg.mot_dataset['SEQS'],
                vis_threshold=0.1,
            )
            mot_accums = []
            for seq in dataset:
                results = seq.load_results(mot_tracker_path)
                mot_accums.append(
                    get_mot_accum(
                        results,
                        seq,
                        use_ignore_regions=self.cfg.mot_evaluator.use_ignore_regions,
                        ignore_iou_thres=self.cfg.mot_evaluator.ignore_iou_thres,
                    )
                )
            if mot_accums:
                evaluate_mot_accums(
                    mot_accums,
                    [str(s) for s in dataset if not s.no_gt],
                    generate_overall=True,
                )

        if any(
            (
                self.cfg.eval_pose_estimation,
                self.cfg.eval_pose_tracking,
                self.cfg.eval_reid_pose_tracking,
            )
        ):
            images = self._images(tracker_state.gt.image_metadatas)

        if self.cfg.eval_pose_estimation:
            # populate dataset
            dataset_cfg = self.cfg.posetrack_dataset.copy()
            dataset_cfg.TRACKERS_FOLDER = os.path.join(
                self.cfg.posetrack_dataset.TRACKERS_FOLDER, "pose_estimation"
            )
            annotations = self._annotations_pose_estimation_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            self._save_json(images, annotations, dataset_cfg.TRACKERS_FOLDER)
            # run evaluator
            self.pose_estimation_evaluator.evaluate(
                [PoseTrack(dataset_cfg)], self.pose_estimation_metrics
            )

        if self.cfg.eval_pose_tracking:
            # populate dataset
            dataset_cfg = self.cfg.posetrack_dataset.copy()
            dataset_cfg.TRACKERS_FOLDER = os.path.join(
                self.cfg.posetrack_dataset.TRACKERS_FOLDER, "pose_tracking"
            )
            annotations = self._annotations_pose_tracking_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            self._save_json(images, annotations, dataset_cfg.TRACKERS_FOLDER)
            dataset_cfg.TRACKERS_FOLDER = os.path.abspath(dataset_cfg.TRACKERS_FOLDER)
            # run evaluator
            self.pose_tracking_evaluator.evaluate(
                [PoseTrack(dataset_cfg)], self.pose_tracking_metrics
            )

        if self.cfg.eval_reid_pose_tracking:
            # populate dataset
            dataset_cfg = self.cfg.posetrack_dataset.copy()
            dataset_cfg.TRACKERS_FOLDER = os.path.join(
                self.cfg.posetrack_dataset.TRACKERS_FOLDER, "reid_pose_tracking"
            )
            annotations = self._annotations_reid_pose_tracking_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            self._save_json(images, annotations, dataset_cfg.TRACKERS_FOLDER)
            # run evaluator
            self.reid_pose_tracking_evaluator.evaluate(
                [PoseTrack(dataset_cfg)], self.reid_pose_tracking_metrics
            )

    # MOT helper functions
    def _mot_encoding(self, predictions, image_metadatas):
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            predictions.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
        )
        len_before_drop = len(df)
        df.dropna(
            subset=["video_name", "frame", "track_id", "bbox_ltwh", "keypoints_xyc",],
            how="any",
            inplace=True,
        )
        print("Dropped {} rows with NA values".format(len_before_drop - len(df)))
        df["bb_left"] = df["bbox_ltwh"].apply(lambda x: x[0])
        df["bb_top"] = df["bbox_ltwh"].apply(lambda x: x[1])
        df["bb_width"] = df["bbox_ltwh"].apply(lambda x: x[2])
        df["bb_height"] = df["bbox_ltwh"].apply(lambda x: x[3])
        df["conf"] = df["keypoints_xyc"].apply(lambda x: np.mean(x[:, 2]))
        df = df.assign(x=-1, y=-1, z=-1)
        return df

    def _save_mot(self, mot_df, save_path):
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        videos_names = mot_df["video_name"].unique()
        for video_name in videos_names:
            file_path = os.path.join(save_path, f"{video_name}.txt")
            file_df = mot_df[mot_df["video_name"] == video_name].copy()
            file_df.sort_values(by="frame", inplace=True)
            file_df[
                [
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
                ]
            ].to_csv(
                file_path, header=False, index=False,
            )

    # PoseTrack helper functions
    def _images(self, image_metadatas):
        image_metadatas.dropna(
            subset=["video_name", "file_path", "id", "frame",], how="any", inplace=True,
        )
        image_metadatas.rename(columns={"file_path": "file_name"}, inplace=True)
        image_metadatas["frame_id"] = image_metadatas["id"]

        images = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            images_by_video = image_metadatas[
                image_metadatas["video_name"] == video_name
            ]
            images[video_name] = images_by_video[
                ["file_name", "id", "frame_id"]
            ].to_dict("records")
        return images

    # TODO fuse different annotations functions
    def _annotations_pose_estimation_eval(self, predictions, image_metadatas):
        predictions = predictions.copy()  # FIXME is it required ?
        predictions.dropna(
            subset=["keypoints_xyc", "bbox_ltwh", "image_id",], how="any", inplace=True,
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", "bbox_ltwh": "bbox"}, inplace=True,
        )
        predictions["scores"] = predictions["keypoints"].apply(
            lambda x: x[:, 2]
        )  # FIXME should not be there
        predictions["track_id"] = predictions["id"]
        predictions["person_id"] = predictions["id"]

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[image_metadatas["video_name"] == video_name].id
            predictions_by_video = predictions[predictions["image_id"].isin(image_ids)]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("records")
        return annotations

    def _annotations_pose_tracking_eval(self, predictions, image_metadatas):
        predictions = predictions.copy()  # FIXME is it required ?
        predictions.dropna(
            subset=["keypoints_xyc", "bbox_ltwh", "image_id", "track_id",],
            how="any",
            inplace=True,
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", "bbox_ltwh": "bbox"}, inplace=True,
        )
        predictions["scores"] = predictions["keypoints"].apply(
            lambda x: x[:, 2]
        )  # FIXME should not be there
        predictions["track_id"] = predictions["track_id"].astype(int)
        predictions["person_id"] = predictions["id"]

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[image_metadatas["video_name"] == video_name].id
            predictions_by_video = predictions[predictions["image_id"].isin(image_ids)]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("records")
        return annotations

    def _annotations_reid_pose_tracking_eval(self, predictions, image_metadatas):
        predictions = predictions.copy()  # FIXME is it required ?
        predictions.dropna(
            subset=["keypoints_xyc", "bbox_ltwh", "image_id", "track_id", "person_id"],
            how="any",
            inplace=True,
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", "bbox_ltwh": "bbox"}, inplace=True,
        )
        predictions["scores"] = predictions["keypoints"].apply(
            lambda x: x[:, 2]
        )  # FIXME should be somewhere else
        predictions["track_id"] = predictions["track_id"].astype(int)
        predictions["person_id"] = predictions["person_id"].astype(int)

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[image_metadatas["video_name"] == video_name].id
            predictions_by_video = predictions[predictions["image_id"].isin(image_ids)]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("records")
        return annotations

    def _save_json(self, images, annotations, path):
        os.makedirs(path, exist_ok=True)
        for video_name in images.keys():
            file_path = os.path.join(path, f"{video_name}.json")
            with open(file_path, "w") as f:
                json.dump(
                    {
                        "images": images[video_name],
                        "annotations": annotations[video_name],
                    },
                    f,
                    cls=PoseTrack21Encoder,
                )


class PoseTrack21Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return json.JSONEncoder.default(self, obj)
