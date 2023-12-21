import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import pbtrack

root_dir = Path(pbtrack.__file__).parents[1]
sys.path.append(str((root_dir / "plugins/eval/TrackEval").resolve()))
import trackeval  # noqa: E402

from pbtrack.core import Evaluator as EvaluatorBase
from pbtrack.utils import wandb

log = logging.getLogger(__name__)


class DanceTrackEvaluator(EvaluatorBase):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    def run(self, tracker_state):
        log.info("Starting evaluation on DanceTrack")
        image_metadatas = (
            tracker_state.image_metadatas.merge(tracker_state.video_metadatas["name"], left_on="video_id",
                                                right_on="id", ).set_index(tracker_state.image_metadatas.index).rename(
                columns={"name": "video_name"}))
        seqs = list(tracker_state.video_metadatas.name)
        bbox_column = self.cfg.bbox_column_for_eval

        if self.cfg.eval_mot:
            # Bounding box evaluation
            # AP
            bbox_map = self.compute_bbox_map(tracker_state.detections_pred, tracker_state.detections_gt,
                                             tracker_state.image_metadatas, bbox_column, )
            map(bbox_map.pop, ["map_per_class", "mar_100_per_class"])
            headers = bbox_map.keys()
            data = [np.round(100 * bbox_map[name].item(), 2) for name in headers]
            log.info("MOT - bbox mAP\n" + tabulate([data], headers=headers, tablefmt="plain"))
            wandb.log(bbox_map, "DanceTrack/bbox/AP")

            trackers_folder = self.cfg.mot_trackers_folder
            mot_df = self._mot_encoding(tracker_state.detections_pred, image_metadatas, bbox_column)
            self._save_mot(mot_df, trackers_folder)

            # copied from TrackEval/scripts/run_mot_challenge.py
            default_eval_config = trackeval.Evaluator.get_default_eval_config()
            default_eval_config["USE_PARALLEL"] = self.cfg.use_parallel
            default_eval_config["NUM_PARALLEL_CORES"] = max(1, self.cfg.num_parallel_cores)
            default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
            config = {**default_eval_config, **default_dataset_config,
                      **default_metrics_config}  # Merge default configs
            eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
            dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
            dataset_config["GT_FOLDER"] = self.cfg.mot_gt_folder
            dataset_config["TRACKERS_FOLDER"] = trackers_folder
            metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

            # Run code
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config, seqs)]
            metrics_list = []
            for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                           trackeval.metrics.VACE]:
                if metric.get_name() in metrics_config['METRICS']:
                    metrics_list.append(metric(metrics_config))
            if len(metrics_list) == 0:
                raise Exception('No metrics selected for evaluation')
            output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
            wandb.log(output_res, "DanceTrack/bbox/MOT")

    def check_if_tracklet(self, detections):
        detections = detections.sort_values(by=["video_id", "track_id"])

        tracklet_lengths = (detections.groupby(["video_id", "track_id"]).size().reset_index(name="tracklet_len"))

        detections = detections.merge(tracklet_lengths, on=["video_id", "track_id"])
        detections["is_tracklet"] = detections["tracklet_len"].apply(lambda x: x >= self.cfg.min_tracklet_length)

        return detections

    # MOT helper functions
    def _mot_encoding(self, detections_pred, image_metadatas, bbox_column):
        detections_pred = detections_pred.copy()
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(image_metadatas.reset_index(drop=True), detections_pred.reset_index(drop=True), left_on="id",
                      right_on="image_id", suffixes=('', '_y'))
        len_before_drop = len(df)
        df.dropna(subset=["video_name", "frame", "track_id", bbox_column, ], how="any", inplace=True, )
        # drop detections that are in ignored regions
        df = df[df.ignored == False]
        df = self.check_if_tracklet(df)
        df = df[df.is_tracklet == True]
        if len_before_drop != len(df):
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
            if not file_df.empty:
                file_df.sort_values(by="frame", inplace=True)
                file_df[["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "bbox_conf", "x", "y",
                         "z", ]].to_csv(file_path, header=False, index=False, )
            else:
                open(file_path, "w").close()

    @staticmethod
    def compute_bbox_map(detections_pred, detections_gt, metadatas, bbox_column):
        images_ids = metadatas[metadatas.is_labeled].index
        detections_pred = detections_pred[detections_pred.ignored == False]
        metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", num_classes=1)
        preds = []
        targets = []
        for image_id in images_ids:
            targets_by_image = detections_gt[detections_gt["image_id"] == image_id]
            if not targets_by_image.empty:
                targets.append({"boxes": torch.tensor(np.vstack(targets_by_image.bbox_ltwh.values).astype(float)),
                                "labels": torch.tensor(targets_by_image.category_id.values), })
                preds_by_image = detections_pred[detections_pred["image_id"] == image_id]
                if not preds_by_image.empty:
                    preds.append({"boxes": torch.tensor(np.vstack(preds_by_image[bbox_column].values).astype(float)),
                                  "scores": torch.tensor(preds_by_image.bbox_conf.values.astype(float)),
                                  "labels": torch.tensor(preds_by_image.category_id.values.astype(int)), })
                else:
                    preds.append({"boxes": torch.tensor([]), "scores": torch.tensor([]), "labels": torch.tensor([]), })
        metric.update(preds, targets)
        return metric.compute()
