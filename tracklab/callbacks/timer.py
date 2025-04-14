import logging
import time
from datetime import timedelta
from typing import Optional, Any

import pandas as pd
from torch.utils.data import DataLoader
from tracklab.callbacks import Callback

log = logging.getLogger(__name__)


class Timer(Callback):
    def __init__(self, **kwargs):
        self.start_times = {}

    def on_dataset_track_start(self, engine: "TrackingEngine"):
        self.start_times["dataset"] = time.perf_counter()

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        time_delta = timedelta(seconds=time.perf_counter() - self.start_times["dataset"])
        log.info(f"Dataset time : {time_delta}")

    def on_video_loop_start(
        self, engine: "TrackingEngine", video_metadata: pd.Series, video_idx: int, index: int
    ):
        self.start_times["video"] = time.perf_counter()

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        end = time.perf_counter()
        time_delta = timedelta(seconds=end - self.start_times["video"])
        frames = len(image_pred)
        fps = frames / (end - self.start_times["video"])
        log.info(f"Video time : {time_delta}, FPS : {fps}")

    def on_module_start(self, engine: "TrackingEngine", task: str, dataloader: DataLoader):
        self.start_times[task] = time.perf_counter()

    def on_module_end(self, engine: "TrackingEngine", task: str, detections: pd.DataFrame):
        end = time.perf_counter()
        time_delta = timedelta(seconds=end - self.start_times[task])
        frames = len(detections.image_id.unique())
        fps = frames / (end-self.start_times[task])
        log.info(f"Module {task} time : {time_delta}, FPS : {fps}")
