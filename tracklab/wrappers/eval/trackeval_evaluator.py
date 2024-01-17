import os
import numpy as np
import pandas as pd
import logging
import trackeval
import wandb

from pathlib import Path
from tabulate import tabulate
from tracklab.core import Evaluator as EvaluatorBase

log = logging.getLogger(__name__)


class TrackEvalEvaluator(EvaluatorBase):
    """
    Evaluator using the TrackEval library (https://github.com/JonathonLuiten/TrackEval).
    Save on disk the tracking predictions and ground truth in MOT Challenge format and run the evaluation by calling TrackEval.
    """
    def __init__(self, cfg, eval_set, trackeval_dataset_class, *args, **kwargs):
        self.eval_set = eval_set
        self.cfg = cfg
        self.trackeval_dataset_class = getattr(trackeval.datasets, trackeval_dataset_class)  # FIXME move elsewhere?

    def run(self, tracker_state):
        log.info("Starting evaluation using TrackEval library (https://github.com/JonathonLuiten/TrackEval)")

        tracker_name = 'tracklab'

        # save predictions in MOT Challenge format
        pred_save_path = Path(self.cfg.dataset.TRACKERS_FOLDER) / f"{self.trackeval_dataset_class.__name__}-{self.eval_set}" / tracker_name
        save_in_mot_challenge_format(tracker_state.detections_pred,
                                     tracker_state.image_metadatas,
                                     tracker_state.video_metadatas,
                                     pred_save_path,
                                     self.cfg.bbox_column_for_eval)

        log.info("Tracking predictions saved in MOT Challenge format in {}".format(pred_save_path))

        if len(tracker_state.detections_gt) == 0:
            log.info(
                f"Stopping evaluation because the current split ({self.eval_set}) has no ground truth detections.")
            return

        # save ground truth in MOT Challenge format  # FIXME remove
        save_in_mot_challenge_format(tracker_state.detections_gt,
                                     tracker_state.image_metadatas,
                                     tracker_state.video_metadatas,
                                     Path(self.cfg.dataset.GT_FOLDER) / f"{self.trackeval_dataset_class.__name__}-{self.eval_set}",
                                     self.cfg.bbox_column_for_eval)

        log.info("Tracking ground truth saved in MOT Challenge format in {}".format(pred_save_path))

        # Build dataset
        dataset_config = self.trackeval_dataset_class.get_default_dataset_config()
        dataset_config['SEQ_INFO'] = tracker_state.video_metadatas.set_index('name')['nframes'].to_dict()
        for key, value in self.cfg.dataset.items():
            dataset_config[key] = value
        dataset = self.trackeval_dataset_class(dataset_config)

        # Build metrics
        metrics_config = {'METRICS': set(self.cfg.metrics), 'PRINT_CONFIG': False, 'THRESHOLD': 0.5}
        metrics_list = []
        for metric_name in self.cfg.metrics:
            try:
                metric = getattr(trackeval.metrics, metric_name)
                metrics_list.append(metric(metrics_config))
            except AttributeError:
                log.warning(f'Skipping evaluation for unknown metric: {metric_name}')

        # Build evaluator
        eval_config = trackeval.Evaluator.get_default_eval_config()
        for key, value in self.cfg.eval.items():
            eval_config[key] = value
        evaluator = trackeval.Evaluator(eval_config)

        # Run evaluation
        output_res, output_msg = evaluator.evaluate([dataset], metrics_list)
        
        # Log results
        results = output_res[dataset.get_name()][tracker_name]
        combined_results = results.pop('SUMMARIES')
        wandb.log(combined_results)


def save_in_mot_challenge_format(detections, image_metadatas, video_metadatas, save_folder, bbox_column_for_eval="bbox_ltwh"):
    mot_df = _mot_encoding(detections, image_metadatas, video_metadatas, bbox_column_for_eval)

    save_path = os.path.join(save_folder)
    os.makedirs(save_path, exist_ok=True)

    # MOT Challenge format = <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    # videos_names = mot_df["video_name"].unique()
    for id, video in video_metadatas.iterrows():
        file_path = os.path.join(save_path, f"{video['name']}.txt")
        file_df = mot_df[mot_df["video_id"] == id].copy()
        if file_df["frame"].min() == 0:
            file_df["frame"] = file_df["frame"] + 1  # MOT Challenge format starts at 1
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
                    "category_id",
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


def _mot_encoding(detections, image_metadatas, video_metadatas, bbox_column):
    detections = detections.copy()
    image_metadatas["id"] = image_metadatas.index
    df = pd.merge(
        image_metadatas.reset_index(drop=True),
        detections.reset_index(drop=True),
        left_on="id",
        right_on="image_id",
        suffixes=('', '_y')
    )
    len_before_drop = len(df)
    df.dropna(
        subset=[
            "frame",
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


def _print_results(
    res_combined,
    res_by_video=None,
    scale_factor=1.0,
    title="",
    print_by_video=False,
):
    headers = res_combined.keys()
    data = [
        format_metric(name, res_combined[name], scale_factor)
        for name in headers
    ]
    log.info(f"{title}\n" + tabulate([data], headers=headers, tablefmt="plain"))
    if print_by_video and res_by_video:
        data = []
        for video_name, res in res_by_video.items():
            video_data = [video_name] + [
                format_metric(name, res[name], scale_factor)
                for name in headers
            ]
            data.append(video_data)
        headers = ["video"] + list(headers)
        log.info(
            f"{title} by videos\n"
            + tabulate(data, headers=headers, tablefmt="plain")
        )


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
