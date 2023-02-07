import os
import sys
import json
import numpy as np
import pandas as pd
from tabulate import tabulate

from pbtrack.core.evaluator import Evaluator as EvaluatorBase

import pbtrack
from pathlib import Path
from pbtrack.utils import wandb

root_dir = Path(pbtrack.__file__).parents[1]

sys.path.append(
    str((root_dir / "plugins/eval/PoseTrack21/eval/posetrack21").resolve())
)  # FIXME : ugly
import posetrack21

sys.path.append(
    str((root_dir / "plugins/eval/PoseTrack21/eval/mot").resolve())
)  # FIXME : ugly
from datasets.pt_warper import PTWrapper
from evaluate_mot import get_mot_accum, evaluate_mot_accums


def format_metric(metric_name, metric_value, scale_factor):
    if "TP" in metric_name or "FN" in metric_name or "FP" in metric_name or "TN" in metric_name:
        return int(metric_value)
    else:
        return np.around(metric_value * scale_factor, 2)

class PoseTrack21(EvaluatorBase):
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, tracker_state):
        images = self._images(tracker_state.gt.image_metadatas)
        seqs = list(tracker_state.gt.video_metadatas.name)
        if self.cfg.eval_pose_estimation:
            annotations = self._annotations_pose_estimation_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_estimation"
            )
            self._save_json(images, annotations, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="pose_estim",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            wandb.log(res_combined, "pose", res_by_video)
            print("Pose estimation results: ")
            self._print_results(res_combined, res_by_video, scale_factor=1.0)

        if self.cfg.eval_pose_tracking:
            annotations = self._annotations_tracking_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_tracking"
            )
            self._save_json(images, annotations, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="pose_tracking",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            print("Pose tracking results:")
            self._print_results(res_combined, res_by_video, scale_factor=100)
            wandb.log(res_combined, "posetrack", res_by_video)

        if self.cfg.eval_reid_pose_tracking:
            annotations = self._annotations_reid_pose_tracking_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "reid_pose_tracking"
            )
            self._save_json(images, annotations, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="reid_tracking",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            print("Reid pose tracking results:")
            self._print_results(res_combined, res_by_video, scale_factor=100)
            wandb.log(res_combined, "reid", res_by_video)

        if self.cfg.eval_mot:
            # HOTA
            trackers_folder = self.cfg.mot_trackers_folder
            mot_df = self._mot_encoding(
                tracker_state.predictions, tracker_state.gt.image_metadatas
            )
            self._save_mot(mot_df, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.mot_gt_folder,
                eval_type="posetrack_mot",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            print("Posetrack MOT results (HOTA):")
            self._print_results(res_combined, res_by_video, 100)
            wandb.log(res_combined, "mot", res_by_video)
            # MOTA
            dataset = PTWrapper(
                self.cfg.mot_gt_folder,
                self.cfg.mot.dataset_path,
                seqs,
                vis_threshold=self.cfg.mot.vis_threshold,
            )
            mot_accums = []
            for seq in dataset:
                results = seq.load_results(os.path.join(trackers_folder, "results"))
                mot_accums.append(
                    get_mot_accum(
                        results,
                        seq,
                        use_ignore_regions=self.cfg.mot.use_ignore_regions,
                        ignore_iou_thres=self.cfg.mot.ignore_iou_thres,
                    )
                )
            if mot_accums:
                print("Posetrack mot results (MOTA):")
                str_summary = evaluate_mot_accums(
                    mot_accums,
                    [str(s) for s in dataset if not s.no_gt],
                    generate_overall=True,
                )

    # PoseTrack helper functions
    @staticmethod
    def _images(image_metadatas):
        image_metadatas.dropna(
            subset=[
                "video_name",
                "file_path",
                "id",
                "frame",
            ],
            how="any",
            inplace=True,
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

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_pose_estimation_eval(predictions, image_metadatas):
        predictions = predictions.copy()
        predictions.dropna(
            subset=[
                "keypoints_xyc",
                "bbox_ltwh",
                "image_id",
            ],
            how="any",
            inplace=True,
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", "bbox_ltwh": "bbox"},
            inplace=True,
        )
        if 'scores' not in predictions.columns:
            # 'scores' can already be present if loaded from a json file with external predictions
            # for PoseTrack21 author baselines, not using their provided score induces a big drop in performance
            predictions["scores"] = predictions["keypoints"].apply(lambda x: x[:, 2])
        predictions["track_id"] = predictions["track_id"]
        predictions["person_id"] = predictions["track_id"]

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[image_metadatas["video_name"] == video_name].id
            predictions_by_video = predictions[predictions["image_id"].isin(image_ids)]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("records")
        return annotations

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_tracking_eval(predictions, image_metadatas):
        predictions = predictions.copy()
        predictions.dropna(
            subset=[
                "keypoints_xyc",
                "track_bbox_kf_ltwh",
                "image_id",
                "track_id",
            ],
            how="any",
            inplace=True,
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", "track_bbox_kf_ltwh": "bbox"},
            inplace=True,
        )
        predictions["scores"] = predictions["keypoints"].apply(lambda x: x[:, 2])
        predictions["track_id"] = predictions["track_id"].astype(int)
        predictions["person_id"] = predictions["id"].astype(int)

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[image_metadatas["video_name"] == video_name].id
            predictions_by_video = predictions[predictions["image_id"].isin(image_ids)]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("records")
        return annotations

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_reid_pose_tracking_eval(predictions, image_metadatas):
        predictions = predictions.copy()
        predictions.dropna(
            subset=[
                "keypoints_xyc",
                "track_bbox_kf_ltwh",
                "image_id",
                "track_id",
                "person_id",
            ],
            how="any",
            inplace=True,
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", "track_bbox_kf_ltwh": "bbox"},
            inplace=True,
        )
        predictions["scores"] = predictions["keypoints"].apply(lambda x: x[:, 2])
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

    @staticmethod
    def _save_json(images, annotations, path):
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

    # MOT helper functions
    @staticmethod
    def _mot_encoding(predictions, image_metadatas):
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            predictions.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
        )
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "video_name",
                "frame",
                "track_id",
                "track_bbox_kf_ltwh",
                "keypoints_xyc",
            ],
            how="any",
            inplace=True,
        )
        print("Dropped {} rows with NA values".format(len_before_drop - len(df)))
        df["track_id"] = df["track_id"].astype(int)
        df["bb_left"] = df["track_bbox_kf_ltwh"].apply(lambda x: x[0])
        df["bb_top"] = df["track_bbox_kf_ltwh"].apply(lambda x: x[1])
        df["bb_width"] = df["track_bbox_kf_ltwh"].apply(lambda x: x[2])
        df["bb_height"] = df["track_bbox_kf_ltwh"].apply(lambda x: x[3])
        df = df.assign(x=-1, y=-1, z=-1)
        return df

    @staticmethod
    def _save_mot(mot_df, save_path):
        save_path = os.path.join(save_path, "results")
        os.makedirs(save_path, exist_ok=True)
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
                    "bbox_c",
                    "x",
                    "y",
                    "z",
                ]
            ].to_csv(
                file_path,
                header=False,
                index=False,
            )

    def _print_results(self, res_combined, res_by_video, scale_factor):
        headers = res_combined.keys()
        data = [format_metric(name, res_combined[name], scale_factor) for name in headers]
        print(tabulate([data], headers=headers, tablefmt="pretty"))
        if self.cfg.print_by_video:
            print("By videos:")
            data = []
            for video_name, res in res_by_video.items():
                video_data = [video_name] + [format_metric(name, res[name], scale_factor) for name in headers]
                data.append(video_data)
            headers = ["video"] + list(headers)
            print(tabulate(data, headers=headers, tablefmt="pretty"))


class PoseTrack21Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return json.JSONEncoder.default(self, obj)
