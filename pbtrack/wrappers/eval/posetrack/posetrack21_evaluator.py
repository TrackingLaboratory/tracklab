import os
import json
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate

from pbtrack import Evaluator as EvaluatorBase
from pbtrack.utils import wandb

import posetrack21
import posetrack21_mot
from poseval.eval_helpers import (
    load_data_dir,
    Joint,
    mapmetrics2dict,
    precmetrics2dict,
    recallmetrics2dict,
    motmetrics2dict,
)
from poseval.evaluateAP import evaluateAP
from poseval.evaluateTracking import evaluateTracking
from torchmetrics.detection.mean_ap import MeanAveragePrecision


import logging

log = logging.getLogger(__name__)


def format_metric(metric_name, metric_value, scale_factor):
    if (
        "TP" in metric_name
        or "FN" in metric_name
        or "FP" in metric_name
        or "TN" in metric_name
    ):
        return int(metric_value)
    else:
        return np.around(metric_value * scale_factor, 2)


class PoseTrack21(EvaluatorBase):
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, tracker_state):
        images = self._images(tracker_state.gt.image_metadatas)
        category = self._category(tracker_state.gt.video_metadatas)
        seqs = list(tracker_state.gt.video_metadatas.name)
        bbox_column = self.cfg.bbox_column_for_eval
        eval_pose_on_all = self.cfg.eval_pose_on_all
        log.info("Evaluation on PoseTrack21")
        if self.cfg.eval_pose_estimation:
            annotations = self._annotations_pose_estimation_eval(
                tracker_state.predictions,
                tracker_state.gt.image_metadatas,
                bbox_column,
                eval_pose_on_all,
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_estimation"
            )
            self._save_json(images, annotations, category, trackers_folder)

            log.info("Bbox estimation")
            bbox_map = self.compute_bbox_map(
                tracker_state.predictions,
                tracker_state.gt.detections,
                tracker_state.gt.image_metadatas,
            )
            log.info("Average Precision (AP) metric")
            map(bbox_map.pop, ["map_per_class", "mar_100_per_class"])
            headers = bbox_map.keys()
            data = [np.round(100 * bbox_map[name].item(), 2) for name in headers]
            log.info("\n" + tabulate([data], headers=headers, tablefmt="plain"))
            wandb.log(bbox_map, "PoseTrack21/Bbox mAP")

            log.info("Pose estimation")
            log.info("Average Precision (AP) metric")
            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            apAll, preAll, recAll = evaluateAP(
                gtFramesAll, prFramesAll, "", False, False
            )
            res_combined = mapmetrics2dict(apAll)
            self._print_results(res_combined, scale_factor=1.0)
            wandb.log(res_combined, "PoseTrack21/mAP kp")
            log.info("Precision metric")
            res_combined = precmetrics2dict(preAll)
            self._print_results(res_combined, scale_factor=1.0)
            wandb.log(res_combined, "PoseTrack21/Precision kp")
            log.info("Recall metric")
            res_combined = recallmetrics2dict(recAll)
            self._print_results(res_combined, scale_factor=1.0)
            wandb.log(res_combined, "PoseTrack21/Recall kp")

        if self.cfg.eval_pose_tracking:
            annotations = self._annotations_tracking_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas, bbox_column
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_tracking"
            )
            self._save_json(images, annotations, category, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="pose_tracking",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            log.info("Keypoints tracking results (HOTA)")
            self._print_results(res_combined, res_by_video, scale_factor=100)
            wandb.log(res_combined, "PoseTrack21/HOTA kp", res_by_video)

            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            metricsAll = evaluateTracking(gtFramesAll, prFramesAll, "", False, False)
            metrics = np.zeros([Joint().count + 4, 1])
            for i in range(Joint().count + 1):
                metrics[i, 0] = metricsAll["mota"][0, i]
            metrics[Joint().count + 1, 0] = metricsAll["motp"][0, Joint().count]
            metrics[Joint().count + 2, 0] = metricsAll["pre"][0, Joint().count]
            metrics[Joint().count + 3, 0] = metricsAll["rec"][0, Joint().count]

            log.info("Keypoints tracking results (MOTA)")
            res_combined = motmetrics2dict(metrics)
            self._print_results(res_combined, scale_factor=1.0)
            wandb.log(res_combined, "PoseTrack21/MOTA kp")

        if self.cfg.eval_reid_pose_tracking:
            annotations = self._annotations_reid_pose_tracking_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas, bbox_column
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "reid_pose_tracking"
            )
            self._save_json(images, annotations, category, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="reid_tracking",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            log.info("Keypoints cross-video tracking results")
            self._print_results(res_combined, res_by_video, scale_factor=100)
            wandb.log(res_combined, "PoseTrack21/ReID pose", res_by_video)

        if self.cfg.eval_mot:
            # HOTA
            trackers_folder = self.cfg.mot_trackers_folder
            mot_df = self._mot_encoding(
                tracker_state.predictions, tracker_state.gt.image_metadatas, bbox_column
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
            log.info("Tracking bbox results (HOTA)")
            self._print_results(res_combined, res_by_video, 100)
            wandb.log(res_combined, "PoseTrack21/HOTA bb", res_by_video)
            # MOTA
            dataset = posetrack21_mot.PTWrapper(
                self.cfg.mot_gt_folder,
                self.cfg.mot.dataset_path,
                seqs,
                vis_threshold=self.cfg.mot.vis_threshold,
            )
            mot_accums = []
            for seq in dataset:
                results = seq.load_results(os.path.join(trackers_folder, "results"))
                mot_accums.append(
                    posetrack21_mot.get_mot_accum(
                        results,
                        seq,
                        use_ignore_regions=self.cfg.mot.use_ignore_regions,
                        ignore_iou_thres=self.cfg.mot.ignore_iou_thres,
                    )
                )
            if mot_accums:
                log.info("Posetrack mot results (MOTA)")
                str_summary, summary = posetrack21_mot.evaluate_mot_accums(
                    mot_accums,
                    [str(s) for s in dataset if not s.no_gt],
                    generate_overall=True,
                )
                results_mot_bbox = summary.to_dict(orient="index")
                wandb.log(
                    results_mot_bbox["OVERALL"], "PoseTrack21/MOTA bb", results_mot_bbox
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

    @staticmethod
    def _category(video_metadatas):
        return video_metadatas.categories[0]

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_pose_estimation_eval(
        predictions, image_metadatas, bbox_column, eval_pose_on_all
    ):
        predictions = predictions.copy()
        na_col_to_drop = [
            "keypoints_xyc",
            bbox_column,
            "image_id",
        ]
        if not eval_pose_on_all:
            # If set to false, will evaluate pose estimation only on tracked detections (i.e. detections with a
            # defined 'track_id')
            na_col_to_drop.append("track_id")
        len_before_drop = len(predictions)
        predictions.dropna(
            subset=na_col_to_drop,
            how="any",
            inplace=True,
        )
        log.warning(
            "Dropped {} rows with NA values".format(len_before_drop - len(predictions))
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", bbox_column: "bbox"},
            inplace=True,
        )
        if "scores" not in predictions.columns:
            # 'scores' can already be present if loaded from a json file with external predictions
            # for PoseTrack21 author baselines, not using their provided score induces a big drop in performance
            predictions["scores"] = predictions["keypoints"].apply(lambda x: x[:, 2])
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

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_tracking_eval(predictions, image_metadatas, bbox_column):
        predictions = predictions.copy()
        col_to_drop = [
            "keypoints_xyc",
            bbox_column,
            "image_id",
            "track_id",
        ]
        col_to_drop = [col for col in col_to_drop if col in predictions.columns]
        len_before_drop = len(predictions)
        predictions.dropna(
            subset=col_to_drop,
            how="any",
            inplace=True,
        )
        log.warning(
            "Dropped {} rows with NA values".format(len_before_drop - len(predictions))
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", bbox_column: "bbox"},
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
    def _annotations_reid_pose_tracking_eval(predictions, image_metadatas, bbox_column):
        predictions = predictions.copy()
        len_before_drop = len(predictions)
        predictions.dropna(
            subset=[
                "keypoints_xyc",
                bbox_column,
                "image_id",
                "track_id",
                "person_id",
            ],
            how="any",
            inplace=True,
        )
        log.warning(
            "Dropped {} rows with NA values".format(len_before_drop - len(predictions))
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", bbox_column: "bbox"},
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
    def _save_json(images, annotations, category, path):
        os.makedirs(path, exist_ok=True)
        for video_name in images.keys():
            file_path = os.path.join(path, f"{video_name}.json")
            with open(file_path, "w") as f:
                json.dump(
                    {
                        "images": images[video_name],
                        "annotations": annotations[video_name],
                        "categories": category,
                    },
                    f,
                    cls=PoseTrack21Encoder,
                )

    # MOT helper functions
    @staticmethod
    def _mot_encoding(predictions, image_metadatas, bbox_column):
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
                bbox_column,
                "keypoints_xyc",
            ],
            how="any",
            inplace=True,
        )
        log.warning("Dropped {} rows with NA values".format(len_before_drop - len(df)))
        df["track_id"] = df["track_id"].astype(int)
        df["bb_left"] = df[bbox_column].apply(lambda x: x[0])
        df["bb_top"] = df[bbox_column].apply(lambda x: x[1])
        df["bb_width"] = df[bbox_column].apply(lambda x: x[2])
        df["bb_height"] = df[bbox_column].apply(lambda x: x[3])
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

    def _print_results(self, res_combined, res_by_video=None, scale_factor=1.0):
        headers = res_combined.keys()
        data = [
            format_metric(name, res_combined[name], scale_factor) for name in headers
        ]
        log.info("\n" + tabulate([data], headers=headers, tablefmt="plain"))
        if self.cfg.print_by_video and res_by_video:
            log.info("By videos:")
            data = []
            for video_name, res in res_by_video.items():
                video_data = [video_name] + [
                    format_metric(name, res[name], scale_factor) for name in headers
                ]
                data.append(video_data)
            headers = ["video"] + list(headers)
            log.info("\n" + tabulate(data, headers=headers, tablefmt="plain"))

    @staticmethod
    def compute_bbox_map(predictions, ground_truths, metadatas):
        images_ids = metadatas.id.unique()
        metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", num_classes=1)
        preds = []
        targets = []
        for image_id in images_ids:
            targets_by_image = ground_truths[ground_truths["image_id"] == image_id]
            if not targets_by_image.empty:
                targets.append(
                    {
                        "boxes": torch.tensor(
                            np.vstack(targets_by_image.bbox_ltwh.values).astype(float)
                        ),
                        "labels": torch.tensor(targets_by_image.category_id.values),
                    }
                )
                preds_by_image = predictions[predictions["image_id"] == image_id]
                if not preds_by_image.empty:
                    preds.append(
                        {
                            "boxes": torch.tensor(
                                np.vstack(preds_by_image.bbox_ltwh.values).astype(float)
                            ),
                            "scores": torch.tensor(preds_by_image.bbox_c.values),
                            "labels": torch.tensor(preds_by_image.category_id.values),
                        }
                    )
                else:
                    preds.append(
                        {
                            "boxes": torch.tensor([]),
                            "scores": torch.tensor([]),
                            "labels": torch.tensor([]),
                        }
                    )
        metric.update(preds, targets)
        return metric.compute()


class PoseTrack21Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return json.JSONEncoder.default(self, obj)
