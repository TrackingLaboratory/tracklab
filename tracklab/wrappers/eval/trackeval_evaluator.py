import os
from pathlib import Path

import numpy as np
import pandas as pd
import logging
import trackeval

from tabulate import tabulate
from tracklab.core import Evaluator as EvaluatorBase
from tracklab.utils import wandb


log = logging.getLogger(__name__)


class TrackEvalEvaluator(EvaluatorBase):
    """
    Evaluator using the TrackEval library (https://github.com/JonathonLuiten/TrackEval).
    Save on disk the tracking predictions and ground truth in MOT Challenge format and run the evaluation by calling TrackEval.
    """
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    def run(self, tracker_state):
        log.info("Starting evaluation using TrackEval library (https://github.com/JonathonLuiten/TrackEval)")

        seqs = list(tracker_state.video_metadatas.name)
        split = 'test'
        dataset_name = 'SNMOT'

        # save predictions in MOT Challenge format
        pred_save_path = Path(self.cfg.pred_folder) / f"{dataset_name}-{split}" / "tracklab"
        save_in_mot_challenge_format(tracker_state.detections_pred,
                                     tracker_state.image_metadatas,
                                     tracker_state.video_metadatas,
                                     pred_save_path,
                                     self.cfg.bbox_column_for_eval)

        log.info("Tracking predictions saved in MOT Challenge format in {}".format(pred_save_path))

        if len(tracker_state.detections_gt) == 0:
            log.info(f"Stopping evaluation because the current split ({split}) has no ground truth detections.")
            return

        # save ground truth in MOT Challenge format
        save_in_mot_challenge_format(tracker_state.detections_gt,
                                     tracker_state.image_metadatas,
                                     tracker_state.video_metadatas,
                                     Path(self.cfg.gt_folder) / f"{dataset_name}-{split}",
                                     self.cfg.bbox_column_for_eval)

        log.info("Tracking ground truth saved in MOT Challenge format in {}".format(pred_save_path))

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config['USE_PARALLEL'] = False
        eval_config['NUM_PARALLEL_CORES'] = 8
        eval_config['BREAK_ON_ERROR'] = True  # Raises exception and exits with error
        eval_config['PRINT_RESULTS'] = True
        eval_config['PRINT_ONLY_COMBINED'] = True
        eval_config['PRINT_CONFIG'] = False
        eval_config['TIME_PROGRESS'] = False
        eval_config['DISPLAY_LESS_PROGRESS'] = False
        eval_config['OUTPUT_SUMMARY'] = True
        eval_config['OUTPUT_EMPTY_CLASSES'] = True  # If False, summary files are not output for classes with no detections
        eval_config['OUTPUT_DETAILED'] = True
        eval_config['PLOT_CURVES'] = True

        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'PRINT_CONFIG': False, 'THRESHOLD': 0.5}
        # config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
        #
        # eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        # dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        # metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

        dataset_config['BENCHMARK'] = dataset_name
        dataset_config['GT_FOLDER'] = self.cfg.gt_folder  # Location of GT data
        dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}.txt'  # '{gt_folder}/{seq}/gt/gt.txt'
        dataset_config['TRACKERS_FOLDER'] = self.cfg.pred_folder  # Trackers location
        dataset_config['TRACKER_SUB_FOLDER'] = ''  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        dataset_config['OUTPUT_FOLDER'] = self.cfg.results_folder # Where to save eval results (if None, same as TRACKERS_FOLDER)
        dataset_config['OUTPUT_SUB_FOLDER'] = ''  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        dataset_config['SEQ_INFO'] = tracker_state.video_metadatas.set_index('name')['nframes'].to_dict()
        # dataset_config['TRACKERS_TO_EVAL'] = None  # Filenames of trackers to eval (if None, all in folder)
        dataset_config['CLASSES_TO_EVAL'] = ['pedestrian']  # Valid: ['pedestrian']
        dataset_config['SPLIT_TO_EVAL'] = split  # Valid: 'train', 'test', 'all'
        # dataset_config['INPUT_AS_ZIP'] = False  # Whether tracker input files are zipped
        dataset_config['PRINT_CONFIG'] = False  # Whether to print current config
        dataset_config['DO_PREPROC'] = False  # Whether to perform preprocessing (never done for MOT15)  # TODO ???
        dataset_config['TRACKER_DISPLAY_NAMES'] = None  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        # dataset_config['SEQMAP_FOLDER'] = None  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
        # dataset_config['SEQMAP_FILE'] = None  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
        # dataset_config['SKIP_SPLIT_FOL'] = False  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in

        # Run code
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                       trackeval.metrics.VACE]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        # results = output_res['MotChallenge2DBox']['tracklab']
        # combined_results = results.pop('COMBINED_SEQ')['pedestrian']
        # wandb.log(output_res, "PoseTrack21/bbox/HOTA", res_by_video)

        # evaluator = posetrack21.api.get_api(
        #     pred_folder=self.cfg.pred_folder,
        #     gt_folder=self.cfg.gt_folder,
        #     eval_type="posetrack_mot",
        #     use_parallel=self.cfg.use_parallel,
        #     num_parallel_cores=max(1, self.cfg.num_parallel_cores),
        #     SEQS=seqs,
        # )
        # res_combined, res_by_video = evaluator.eval()
        # res_combined['MotChallenge2DBox']['tracklab']['COMBINED_SEQ']['pedestrian']['HOTA']
        # _print_results(
        #     res_combined['MotChallenge2DBox']['tracklab']['COMBINED_SEQ']['pedestrian'],
        #     res_by_video,
        #     100,
        #     title="MOT - bbox HOTA",
        #     print_by_video=self.cfg.print_by_video,
        # )
        # wandb.log(res_combined, "PoseTrack21/bbox/HOTA", res_by_video)

        # # MOTA
        # dataset = posetrack21_mot.PTWrapper(
        #     self.cfg.gt_folder,
        #     self.cfg.dataset_path,
        #     seqs,
        #     vis_threshold=self.cfg.vis_threshold,
        # )
        # mot_accums = []
        # for seq in dataset:
        #     results = seq.load_results(os.path.join(pred_folder, "results"))
        #     mot_accums.append(
        #         posetrack21_mot.get_mot_accum(
        #             results,
        #             seq,
        #             use_ignore_regions=self.cfg.use_ignore_regions,
        #             ignore_iou_thres=self.cfg.ignore_iou_thres,
        #         )
        #     )
        # if mot_accums:
        #     log.info("MOT - bbox MOTA")
        #     str_summary, summary = posetrack21_mot.evaluate_mot_accums(
        #         mot_accums,
        #         [str(s) for s in dataset if not s.no_gt],
        #         generate_overall=True,
        #     )
        #     results_mot_bbox = summary.to_dict(orient="index")
        #     wandb.log(
        #         results_mot_bbox["OVERALL"],
        #         "PoseTrack21/bbox/MOTA",
        #         results_mot_bbox,
        #     )


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


def _mot_encoding(detections, image_metadatas, video_metadatas, bbox_column):

    # image_metadatas = (
    #     tracker_state.image_metadatas.merge(
    #         tracker_state.video_metadatas["name"],
    #         left_on="video_id",
    #         right_on="id",
    #     )
    #     .set_index(tracker_state.image_metadatas.index)
    #     .rename(columns={"name": "video_name"})
    # )

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
    # drop detections that are in ignored regions
    # df = df[df.ignored == False]
    # df = self.check_if_tracklet(df)
    # df = df[df.is_tracklet == True]
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
