import os
import json
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate

from pbtrack.core import Evaluator as EvaluatorBase
from pbtrack.utils import wandb

from collections import OrderedDict
import motmetrics as mm
from motmetrics.apps.evaluateTracking import compare_dataframes

import logging

log = logging.getLogger(__name__)


class MOT20Evaluator(EvaluatorBase):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    def run(self, tracker_state):
        log.info("Starting evaluation on MOT20")
        image_metadatas = (
            tracker_state.image_metadatas.merge(
                tracker_state.video_metadatas["name"],
                left_on="video_id",
                right_on="id",
            )
            .set_index(tracker_state.image_metadatas.index)
            .rename(columns={"name": "video_name"})
        )
        seqs = list(tracker_state.video_metadatas.name)
        bbox_column = self.cfg.bbox_column_for_eval

        mot_df = self._mot_encoding(
            tracker_state.detections_pred, image_metadatas, bbox_column
        )
        self._save_mot(mot_df, self.cfg.mot_trackers_folder)
        gtfiles = [os.path.join(self.cfg.mot_gt_folder, i, "gt/gt.txt") for i in seqs]
        tsfiles = [
            os.path.join(self.cfg.mot_trackers_folder, "%s.txt" % i) for i in seqs
        ]

        if self.cfg.eval_mot:
            gt = OrderedDict(
                [
                    (
                        seqs[i],
                        (
                            mm.io.loadtxt(f, fmt="mot16"),
                            os.path.join(
                                self.cfg.mot_gt_folder, seqs[i], "seqinfo.ini"
                            ),
                        ),
                    )
                    for i, f in enumerate(gtfiles)
                ]
            )
            ts = OrderedDict(
                [
                    (seqs[i], mm.io.loadtxt(f, fmt="mot16"))
                    for i, f in enumerate(tsfiles)
                ]
            )
            mh = mm.metrics.create()
            accs, analysis, names = compare_dataframes(
                gt, ts, "", 1.0 - 0.5
            )  # special IoU threshold requirement for small targets
            summary = mh.compute_many(
                accs,
                anas=analysis,
                names=names,
                metrics=mm.metrics.motchallenge_metrics,
                generate_overall=True,
            )
            log.info("MOT - bbox MOTA")
            log.info(
                mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names,
                )
            )
            results_mot_bbox = summary.to_dict(orient="index")
            wandb.log(
                results_mot_bbox["OVERALL"],
                "MOT20/bbox/MOTA",
                results_mot_bbox,
            )

    # MOT helper functions
    @staticmethod
    def _mot_encoding(detections_pred, image_metadatas, bbox_column):
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            detections_pred.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
        )
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "track_id",
                bbox_column,
            ],
            how="any",
            inplace=True,
        )
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
