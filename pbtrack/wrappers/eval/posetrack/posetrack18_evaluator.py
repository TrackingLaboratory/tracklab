import os
import json
import torch
import numpy as np
from tabulate import tabulate

from pbtrack import Evaluator as EvaluatorBase
from pbtrack.utils import wandb

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


# FIXME fuse with PoseTrack21
class PoseTrack18(EvaluatorBase):
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, tracker_state):
        images = self._images(tracker_state.gt.image_metadatas)
        category = self._category(tracker_state.gt.video_metadatas)
        seqs = list(tracker_state.gt.video_metadatas.name)
        bbox_column = self.cfg.bbox_column_for_eval
        eval_pose_on_all = self.cfg.eval_pose_on_all
        log.info("Evaluation on PoseTrack18")
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
            wandb.log(bbox_map, "PoseTrack18/Bbox mAP")

            log.info("Pose estimation")
            log.info("Average Precision (AP) metric")
            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            apAll, preAll, recAll = evaluateAP(
                gtFramesAll, prFramesAll, "", False, False
            )
            res_combined = mapmetrics2dict(apAll)
            self._print_results(res_combined, scale_factor=1.0)
            wandb.log(res_combined, "PoseTrack18/mAP kp")
            log.info("Precision metric")
            res_combined = precmetrics2dict(preAll)
            self._print_results(res_combined, scale_factor=1.0)
            wandb.log(res_combined, "PoseTrack18/Precision kp")
            log.info("Recall metric")
            res_combined = recallmetrics2dict(recAll)
            self._print_results(res_combined, scale_factor=1.0)
            wandb.log(res_combined, "PoseTrack18/Recall kp")

        if self.cfg.eval_pose_tracking:
            annotations = self._annotations_tracking_eval(
                tracker_state.predictions, tracker_state.gt.image_metadatas, bbox_column
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_tracking"
            )
            self._save_json(images, annotations, category, trackers_folder)

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
            wandb.log(res_combined, "PoseTrack18/MOTA kp")

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
        predictions.dropna(
            subset=na_col_to_drop,
            how="any",
            inplace=True,
        )
        predictions.rename(
            columns={"keypoints_xyc": "keypoints", bbox_column: "bbox"},
            inplace=True,
        )
        if "scores" not in predictions.columns:
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
        predictions.dropna(
            subset=col_to_drop,
            how="any",
            inplace=True,
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
                    cls=PoseTrack18Encoder,
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


class PoseTrack18Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return json.JSONEncoder.default(self, obj)
