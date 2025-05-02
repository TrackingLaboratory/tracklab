import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tracklab.pipeline import Evaluator as EvaluatorBase
from tracklab.utils import wandb

try:
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
except ImportError:
    posetrack21 = None

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import logging

log = logging.getLogger(__name__)


# FIXME some parts can be cleaned but works for now
class PoseTrack21Evaluator(EvaluatorBase):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    def run(self, tracker_state):
        log.info("Starting evaluation on PoseTrack21")
        image_metadatas = (
            tracker_state.image_metadatas.merge(
                tracker_state.video_metadatas["name"],
                left_on="video_id",
                right_on="id",
            )
            .set_index(tracker_state.image_metadatas.index)
            .rename(columns={"name": "video_name"})
        )
        images = self._images(image_metadatas)
        category = self._category(tracker_state.video_metadatas)
        seqs = list(tracker_state.video_metadatas.name)
        bbox_column = self.cfg.bbox_column_for_eval
        keypoints_column = self.cfg.keypoints_column_for_eval
        eval_pose_on_all = self.cfg.eval_pose_on_all
        if not self.cfg.get("save_eval", True):
            tempdir = Path(tempfile.TemporaryDirectory().name)
            self.cfg.posetrack_trackers_folder = str(tempdir / self.cfg.posetrack_trackers_folder)
            self.cfg.mot_trackers_folder = str(tempdir / self.cfg.mot_trackers_folder)
        if self.cfg.eval_pose_estimation:
            annotations = self._annotations_pose_estimation_eval(
                tracker_state.detections_pred,
                image_metadatas,
                bbox_column,
                keypoints_column,
                eval_pose_on_all,
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_estimation"
            )
            self._save_json(images, annotations, category, trackers_folder)

            # Keypoint evaluation
            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            apAll, preAll, recAll = evaluateAP(
                gtFramesAll, prFramesAll, "", False, False
            )
            res_combined = mapmetrics2dict(apAll)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints average precision",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack21/kp/AP")
            res_combined = precmetrics2dict(preAll)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints precision",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack21/kp/precision")
            res_combined = recallmetrics2dict(recAll)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints recall",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack21/kp/recall")

        if self.cfg.eval_pose_tracking:
            annotations = self._annotations_tracking_eval(
                tracker_state.detections_pred,
                image_metadatas,
                bbox_column,
                keypoints_column,
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_tracking"
            )
            self._save_json(images, annotations, category, trackers_folder)

            # Keypoint tracking evaluation
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="pose_tracking",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=max(1, self.cfg.num_parallel_cores),
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            self._print_results(
                res_combined,
                res_by_video,
                scale_factor=100,
                title="Pose tracking - keypoints HOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack21/kp/HOTA", res_by_video)

            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            metricsAll = evaluateTracking(gtFramesAll, prFramesAll, "", False, False)

            metrics = np.zeros([Joint().count + 7, 1])
            for i in range(Joint().count + 1):
                metrics[i, 0] = metricsAll["mota"][0, i]
            metrics[Joint().count + 1, 0] = metricsAll["motp"][0, Joint().count]
            metrics[Joint().count + 2, 0] = metricsAll["pre"][0, Joint().count]
            metrics[Joint().count + 3, 0] = metricsAll["rec"][0, Joint().count]
            metrics[Joint().count + 4, 0] = metricsAll["num_misses"][0, Joint().count]
            metrics[Joint().count + 5, 0] = metricsAll["num_switches"][0, Joint().count]
            metrics[Joint().count + 6, 0] = metricsAll["num_false_positives"][
                0, Joint().count
            ]
            res_combined = motmetrics2dict(metrics)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose tracking - keypoints MOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack21/kp/MOTA")

        if self.cfg.eval_reid_pose_tracking:
            annotations = self._annotations_reid_pose_tracking_eval(
                tracker_state.detections_pred,
                image_metadatas,
                bbox_column,
                keypoints_column,
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
                num_parallel_cores=max(1, self.cfg.num_parallel_cores),
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            self._print_results(
                res_combined,
                res_by_video,
                scale_factor=100,
                title="Pose tracking cross-video - keypoints HOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack21/kp/ReID", res_by_video)

        if self.cfg.eval_mot:
            # Bounding box evaluation
            # AP
            bbox_map = self.compute_bbox_map(
                tracker_state.detections_pred,
                tracker_state.detections_gt,
                tracker_state.image_metadatas,
                bbox_column,
            )
            map(bbox_map.pop, ["map_per_class", "mar_100_per_class"])
            headers = bbox_map.keys()
            data = [np.round(100 * bbox_map[name].item(), 2) for name in headers]
            log.info(
                "MOT - bbox mAP\n" + tabulate([data], headers=headers, tablefmt="plain")
            )
            wandb.log_metric(bbox_map, "PoseTrack21/bbox/AP")

            # HOTA
            trackers_folder = self.cfg.mot_trackers_folder
            mot_df = self._mot_encoding(
                tracker_state.detections_pred, image_metadatas, bbox_column
            )
            self._save_mot(mot_df, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.mot_gt_folder,
                eval_type="posetrack_mot",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=max(1, self.cfg.num_parallel_cores),
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            self._print_results(
                res_combined,
                res_by_video,
                100,
                title="MOT - bbox HOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack21/bbox/HOTA", res_by_video)

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
                log.info("MOT - bbox MOTA")
                str_summary, summary = posetrack21_mot.evaluate_mot_accums(
                    mot_accums,
                    [str(s) for s in dataset if not s.no_gt],
                    generate_overall=True,
                )
                results_mot_bbox = summary.to_dict(orient="index")
                wandb.log_metric(
                    results_mot_bbox["OVERALL"],
                    "PoseTrack21/bbox/MOTA",
                    results_mot_bbox,
                )

    # PoseTrack helper functions
    @staticmethod
    def _images(image_metadatas):
        len_before_drop = len(image_metadatas)
        image_metadatas["id"] = image_metadatas.index
        image_metadatas.dropna(
            subset=[
                "file_path",
                "id",
                "frame",
            ],
            how="any",
            inplace=True,
        )
        if len_before_drop != len(image_metadatas):
            log.warning(
                "Dropped {} rows with NA values from image metadatas".format(
                    len_before_drop - len(image_metadatas)
                )
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
    def _annotations_pose_estimation_eval(
        self,
        detections_pred,
        image_metadatas,
        bbox_column,
        keypoints_column,
        eval_pose_on_all,
    ):
        detections_pred = detections_pred.copy()
        detections_pred["id"] = detections_pred.index
        na_col_to_drop = [keypoints_column, bbox_column, "image_id"]
        if not eval_pose_on_all:
            # If set to false, will evaluate pose estimation only on tracked detections (i.e. detections with a
            # defined 'track_id')
            na_col_to_drop.append("track_id")
        na_col_to_drop = [
            col for col in na_col_to_drop if col in detections_pred.columns
        ]
        len_before_drop = len(detections_pred)
        detections_pred.dropna(
            subset=na_col_to_drop,
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        detections_pred = detections_pred[detections_pred.ignored == False]
        # drop detections that are not reliable
        detections_pred["enough_vis_kp"] = detections_pred[keypoints_column].apply(
            lambda x: self.has_enough_vis_kp(x)
        )
        detections_pred = detections_pred[detections_pred.enough_vis_kp == True]
        detections_pred = self.check_if_tracklet(detections_pred)
        detections_pred = detections_pred[detections_pred.is_tracklet == True]
        if len_before_drop != len(detections_pred):
            log.warning(
                "Dropped {} detections".format(len_before_drop - len(detections_pred))
            )
        detections_pred[keypoints_column] = detections_pred[keypoints_column].apply(
            lambda x: self.remove_not_visible_kps(x)
        )
        detections_pred["scores"] = detections_pred[keypoints_column].apply(
            lambda x: x[:, 2]
        )
        detections_pred["track_id"] = detections_pred["track_id"].astype(int)
        detections_pred["person_id"] = detections_pred["id"].astype(int)
        detections_pred.rename(
            columns={keypoints_column: "keypoints", bbox_column: "bbox"},
            inplace=True,
        )

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[
                image_metadatas["video_name"] == video_name
            ].index
            predictions_by_video = detections_pred[
                detections_pred["image_id"].isin(image_ids)
            ]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("records")
        return annotations

    def remove_not_visible_kps(self, x):
        x[x[:, 2] < self.cfg.vis_kp_threshold, :] = 0.0
        return x

    def has_enough_vis_kp(self, x):
        return (
            x[x[:, 2] > self.cfg.vis_kp_threshold, :].shape[0]
            >= self.cfg.min_num_vis_kp
        )

    def check_if_tracklet(self, detections):
        detections = detections.sort_values(by=["video_id", "track_id"])

        tracklet_lengths = (
            detections.groupby(["video_id", "track_id"])
            .size()
            .reset_index(name="tracklet_len")
        )

        detections = detections.merge(tracklet_lengths, on=["video_id", "track_id"])
        detections["is_tracklet"] = detections["tracklet_len"].apply(
            lambda x: x >= self.cfg.min_tracklet_length
        )

        return detections

    # FIXME fuse different annotations functions
    def _annotations_tracking_eval(
        self, detections_pred, image_metadatas, bbox_column, keypoints_column
    ):
        detections_pred = detections_pred.copy()
        detections_pred["id"] = detections_pred.index
        na_col_to_drop = [
            keypoints_column,
            bbox_column,
            "image_id",
            "track_id",
        ]
        na_col_to_drop = [
            col for col in na_col_to_drop if col in detections_pred.columns
        ]
        len_before_drop = len(detections_pred)
        detections_pred.dropna(
            subset=na_col_to_drop,
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        detections_pred = detections_pred[detections_pred.ignored == False]
        # drop detections that are not reliable
        detections_pred["enough_vis_kp"] = detections_pred[keypoints_column].apply(
            lambda x: self.has_enough_vis_kp(x)
        )
        detections_pred = detections_pred[detections_pred.enough_vis_kp == True]
        detections_pred = self.check_if_tracklet(detections_pred)
        detections_pred = detections_pred[detections_pred.is_tracklet == True]
        if len_before_drop != len(detections_pred):
            log.warning(
                "Dropped {} detections".format(len_before_drop - len(detections_pred))
            )
        detections_pred[keypoints_column] = detections_pred[keypoints_column].apply(
            lambda x: self.remove_not_visible_kps(x)
        )
        detections_pred["scores"] = detections_pred[keypoints_column].apply(
            lambda x: x[:, 2]
        )
        detections_pred["track_id"] = detections_pred["track_id"].astype(int)
        detections_pred["person_id"] = detections_pred["id"].astype(int)
        detections_pred.rename(
            columns={keypoints_column: "keypoints", bbox_column: "bbox"},
            inplace=True,
        )

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[
                image_metadatas["video_name"] == video_name
            ].index
            predictions_by_video = detections_pred[
                detections_pred["image_id"].isin(image_ids)
            ]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "person_id", "track_id"]
            ].to_dict("records")
        return annotations

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_reid_pose_tracking_eval(
        detections_pred, image_metadatas, bbox_column, keypoints_column
    ):
        detections_pred = detections_pred.copy()
        len_before_drop = len(detections_pred)
        detections_pred.dropna(
            subset=[
                keypoints_column,
                bbox_column,
                "image_id",
                "track_id",
                "person_id",
            ],
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        detections_pred.drop(
            detections_pred[detections_pred.ignored].index, inplace=True
        )
        if len_before_drop != len(detections_pred):
            log.warning(
                "Dropped {} rows with NA values".format(
                    len_before_drop - len(detections_pred)
                )
            )
        detections_pred.rename(
            columns={keypoints_column: "keypoints", bbox_column: "bbox"},
            inplace=True,
        )
        detections_pred["scores"] = detections_pred["keypoints"].apply(
            lambda x: x[:, 2]
        )
        detections_pred["track_id"] = detections_pred["track_id"].astype(int)
        detections_pred["person_id"] = detections_pred["person_id"].astype(int)

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[
                image_metadatas["video_name"] == video_name
            ].index
            predictions_by_video = detections_pred[
                detections_pred["image_id"].isin(image_ids)
            ]
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
                    cls=PoseTrack21Evaluator.PoseTrackEncoder,
                    sort_keys=True,
                    indent=4,
                )

    # MOT helper functions
    def _mot_encoding(self, detections_pred, image_metadatas, bbox_column):
        detections_pred = detections_pred.copy()
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            detections_pred.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
            suffixes=('', '_y')
        )
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "video_name",
                "frame",
                "track_id",
                bbox_column,
            ],
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        df = df[df.ignored == False]
        df = self.check_if_tracklet(df)
        df = df[df.is_tracklet == True]
        if len_before_drop != len(df):
            log.warning(
                "Dropped {} rows with NA values".format(len_before_drop - len(df))
            )
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
            if not file_df.empty:
                file_df.sort_values(by="frame", inplace=True)
                file_df[
                    [
                        "frame",
                        "track_id",
                        "bb_left",
                        "bb_top",
                        "bb_width",
                        "bb_height",
                        "bbox_conf",
                        "x",
                        "y",
                        "z",
                    ]
                ].to_csv(
                    file_path,
                    header=False,
                    index=False,
                )
            else:
                open(file_path, "w").close()

    @staticmethod
    def format_metric(metric_name, metric_value, scale_factor):
        if (
            "TP" in metric_name
            or "FN" in metric_name
            or "FP" in metric_name
            or "TN" in metric_name
        ):
            if metric_name == "MOTP":
                return np.around(metric_value * scale_factor, 3)
            return int(metric_value)
        else:
            return np.around(metric_value * scale_factor, 3)

    @staticmethod
    def _print_results(
        res_combined,
        res_by_video=None,
        scale_factor=1.0,
        title="",
        print_by_video=False,
    ):
        headers = res_combined.keys()
        data = [
            PoseTrack21Evaluator.format_metric(name, res_combined[name], scale_factor)
            for name in headers
        ]
        log.info(f"{title}\n" + tabulate([data], headers=headers, tablefmt="plain"))
        if print_by_video and res_by_video:
            data = []
            for video_name, res in res_by_video.items():
                video_data = [video_name] + [
                    PoseTrack21Evaluator.format_metric(name, res[name], scale_factor)
                    for name in headers
                ]
                data.append(video_data)
            headers = ["video"] + list(headers)
            log.info(
                f"{title} by videos\n"
                + tabulate(data, headers=headers, tablefmt="plain")
            )

    @staticmethod
    def compute_bbox_map(detections_pred, detections_gt, metadatas, bbox_column):
        images_ids = metadatas[metadatas.is_labeled].index
        detections_pred = detections_pred[detections_pred.ignored == False]
        metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
        preds = []
        targets = []
        for image_id in images_ids:
            targets_by_image = detections_gt[detections_gt["image_id"] == image_id]
            if not targets_by_image.empty:
                targets.append(
                    {
                        "boxes": torch.tensor(
                            np.vstack(targets_by_image.bbox_ltwh.values).astype(float)
                        ),
                        "labels": torch.tensor(targets_by_image.category_id.values),
                    }
                )
                preds_by_image = detections_pred[
                    detections_pred["image_id"] == image_id
                ]
                if not preds_by_image.empty:
                    preds.append(
                        {
                            "boxes": torch.tensor(
                                np.vstack(preds_by_image[bbox_column].values).astype(
                                    float
                                )
                            ),
                            "scores": torch.tensor(
                                preds_by_image.bbox_conf.values.astype(float)
                            ),
                            "labels": torch.tensor(
                                preds_by_image.category_id.values.astype(int)
                            ),
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

    class PoseTrackEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.flatten().tolist()
            return json.JSONEncoder.default(self, obj)
