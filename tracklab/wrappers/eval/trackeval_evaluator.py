import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import trackeval
from tabulate import tabulate
from tracklab.pipeline import Evaluator as EvaluatorBase

log = logging.getLogger(__name__)


class TrackEvalEvaluator(EvaluatorBase):
    """
    Evaluator using the TrackEval library (https://github.com/JonathonLuiten/TrackEval).
    Save on disk the tracking predictions and ground truth in MOT Challenge format and run the evaluation by calling TrackEval.
    """
    def __init__(self, cfg, eval_set, show_progressbar, dataset_path, tracking_dataset, *args, **kwargs):
        self.cfg = cfg
        self.tracking_dataset = tracking_dataset
        self.eval_set = eval_set
        self.trackeval_dataset_name = type(self.tracking_dataset).__name__
        self.trackeval_dataset_class = getattr(trackeval.datasets, cfg.dataset.dataset_class)
        self.show_progressbar = show_progressbar
        self.dataset_path = dataset_path

    def run(self, tracker_state):
        log.info("Starting evaluation using TrackEval library (https://github.com/JonathonLuiten/TrackEval)")

        tracker_name = 'tracklab'
        save_classes = self.trackeval_dataset_class.__name__ != 'MotChallenge2DBox'

        # Save predictions
        pred_save_path = Path(self.cfg.dataset.TRACKERS_FOLDER) / f"{self.trackeval_dataset_name}-{self.eval_set}" / tracker_name
        self.tracking_dataset.save_for_eval(
            tracker_state.detections_pred,
            tracker_state.image_metadatas,
            tracker_state.video_metadatas,
            pred_save_path,
            self.cfg.bbox_column_for_eval,
            save_classes,  # do not use classes for MOTChallenge2DBox
            is_ground_truth=False,
        )

        log.info(
            f"Tracking predictions saved in {self.trackeval_dataset_name} format in {pred_save_path}")

        if tracker_state.detections_gt is None or len(tracker_state.detections_gt) == 0:
            log.warning(
                f"Stopping evaluation because the current split ({self.eval_set}) has no ground truth detections.")
            return

        # Save ground truth
        gt_save_path = Path(self.cfg.dataset.GT_FOLDER) / f"{self.trackeval_dataset_name}-{self.eval_set}"
        if self.cfg.save_gt:
            self.tracking_dataset.save_for_eval(
                tracker_state.detections_gt,
                tracker_state.image_metadatas,
                tracker_state.video_metadatas,
                gt_save_path,
                self.cfg.bbox_column_for_eval,
                True,
                is_ground_truth=True
            )

        log.info(
            f"Tracking ground truth saved in {self.trackeval_dataset_name} format in {gt_save_path}")

        # Build TrackEval dataset
        dataset_config = self.trackeval_dataset_class.get_default_dataset_config()
        dataset_config['SEQ_INFO'] = tracker_state.video_metadatas.set_index('name')['nframes'].to_dict()
        dataset_config['BENCHMARK'] = self.trackeval_dataset_name  # required for trackeval.datasets.MotChallenge2DBox
        for key, value in self.cfg.dataset.items():
            dataset_config[key] = value

        if not self.cfg.save_gt:
            dataset_config['GT_FOLDER'] = self.dataset_path  # Location of GT data
            dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/Labels-GameState.json'  # '{gt_folder}/{seq}/gt/gt.txt'
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
            if key == "NUM_PARALLEL_CORES":
                value = max(1, int(value))
            eval_config[key] = value
        evaluator = trackeval.Evaluator(eval_config)

        # Run evaluation
        with redirect_stdout(io.StringIO()) as stream:
            output_res, output_msg = evaluator.evaluate([dataset], metrics_list, show_progressbar=self.show_progressbar)
        printed_results = stream.getvalue()
        log.info(printed_results)

        # Log results
        results = output_res[dataset.get_name()][tracker_name]
        # if the dataset has the process_trackeval_results method, use it to process the results
        if hasattr(self.tracking_dataset, 'process_trackeval_results'):
            self.tracking_dataset.process_trackeval_results(results, dataset_config, eval_config)

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

