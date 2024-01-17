import os
import numpy as np
import pandas as pd
import logging
import trackeval
import wandb

from pathlib import Path
from tabulate import tabulate
from tracklab.core import Evaluator as EvaluatorBase
from tracklab.wrappers.eval.soccernet.soccernet_2D_box import SoccerNet2DBox

log = logging.getLogger(__name__)


class TrackEvalEvaluator(EvaluatorBase):
    """
    Evaluator using the TrackEval library (https://github.com/JonathonLuiten/TrackEval).
    Save on disk the tracking predictions and ground truth in MOT Challenge format and run the evaluation by calling TrackEval.
    """
    def __init__(self, cfg, eval_set, *args, **kwargs):
        self.eval_set = eval_set
        self.cfg = cfg

    def run(self, tracker_state):
        log.info("Starting evaluation using TrackEval library (https://github.com/JonathonLuiten/TrackEval)")

        dataset_name = 'SNMOT'

        # save predictions in MOT Challenge format
        pred_save_path = Path(self.cfg.dataset.trackers_folder) / f"{dataset_name}-{self.eval_set}" / "tracklab"
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

        # save ground truth in MOT Challenge format
        save_in_mot_challenge_format(tracker_state.detections_gt,
                                     tracker_state.image_metadatas,
                                     tracker_state.video_metadatas,
                                     Path(self.cfg.dataset.gt_folder) / f"{dataset_name}-{self.eval_set}",
                                     self.cfg.bbox_column_for_eval)

        log.info("Tracking ground truth saved in MOT Challenge format in {}".format(pred_save_path))

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config['USE_PARALLEL'] = self.cfg.eval.use_parallel
        eval_config['NUM_PARALLEL_CORES'] = self.cfg.eval.num_parallel_cores
        eval_config['BREAK_ON_ERROR'] = self.cfg.eval.break_on_error  # Raises exception and exits with error
        eval_config['PRINT_RESULTS'] = self.cfg.eval.print_results
        eval_config['PRINT_ONLY_COMBINED'] = self.cfg.eval.print_only_combined
        eval_config['PRINT_CONFIG'] = self.cfg.eval.print_config
        eval_config['TIME_PROGRESS'] = self.cfg.eval.time_progress
        eval_config['DISPLAY_LESS_PROGRESS'] = self.cfg.eval.display_less_progress
        eval_config['OUTPUT_SUMMARY'] = self.cfg.eval.output_summary
        eval_config['OUTPUT_EMPTY_CLASSES'] = self.cfg.eval.output_empty_classes  # If False, summary files are not output for classes with no detections
        eval_config['OUTPUT_DETAILED'] = self.cfg.eval.output_detailed
        eval_config['PLOT_CURVES'] = self.cfg.eval.plot_curves

        dataset_config = SoccerNet2DBox.get_default_dataset_config()
        metrics_config = {'METRICS': self.cfg.metrics, 'PRINT_CONFIG': False, 'THRESHOLD': 0.5}

        dataset_config['BENCHMARK'] = dataset_name
        dataset_config['GT_FOLDER'] = self.cfg.dataset.gt_folder  # Location of GT data
        dataset_config['GT_LOC_FORMAT'] = self.cfg.dataset.gt_loc_format  # '{gt_folder}/{seq}/gt/gt.txt'
        dataset_config['TRACKERS_FOLDER'] = self.cfg.dataset.trackers_folder  # Trackers location
        dataset_config['TRACKER_SUB_FOLDER'] = self.cfg.dataset.tracker_sub_folder  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        dataset_config['OUTPUT_FOLDER'] = self.cfg.dataset.output_folder # Where to save eval results (if None, same as TRACKERS_FOLDER)
        dataset_config['OUTPUT_SUB_FOLDER'] = self.cfg.dataset.output_sub_folder  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        dataset_config['SEQ_INFO'] = tracker_state.video_metadatas.set_index('name')['nframes'].to_dict()
        dataset_config['CLASSES_TO_EVAL'] = ['pedestrian']  # Valid: ['pedestrian']  # TODO
        dataset_config['SPLIT_TO_EVAL'] = self.cfg.dataset.split_to_eval  # Valid: 'train', 'test', 'all'
        dataset_config['PRINT_CONFIG'] = self.cfg.dataset.print_config  # Whether to print current config
        dataset_config['DO_PREPROC'] = self.cfg.dataset.do_preproc  # Whether to perform preprocessing (never done for MOT15)  # TODO ???
        dataset_config['TRACKER_DISPLAY_NAMES'] = self.cfg.dataset.tracker_display_names  # Names of trackers to display, if None: TRACKERS_TO_EVAL

        # Run code
        evaluator = trackeval.Evaluator(eval_config)
        dataset = SoccerNet2DBox(dataset_config)
        # TODO before moving this to soccernet trackeval class, make sure static class creation will work, i.e. empty classes are not processed and do not slow down process
        classes_to_id = {category["name"]: category["id"]+14 for category in tracker_state.video_metadatas.categories.iloc[0]}
        classes_to_id.update({'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                       'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                       'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13})
        dataset.valid_classes = classes_to_id.keys()
        dataset.class_list = classes_to_id.keys()
        dataset.class_name_to_class_id = classes_to_id

        dataset.valid_class_numbers = list(dataset.class_name_to_class_id.values())
        dataset.should_classes_combine = True  # FIXME
        dataset.use_super_categories = False
        dataset_list = [dataset]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                       trackeval.metrics.VACE]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        results = output_res['SoccerNet2DBox']['tracklab']
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
