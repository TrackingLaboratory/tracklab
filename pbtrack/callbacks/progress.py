from typing import Any, Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

import pbtrack
from . import Callback
from ..datastruct import Detections
from ..engine import TrackingEngine


class Progressbar(Callback):
    def __init__(self):
        self.pbar: Optional[tqdm] = None
        self.task_pbars = {}

    def on_dataset_track_start(self, engine: "TrackingEngine"):
        total = len(engine.video_metadatas)
        self.pbar = tqdm(total=total, desc="Tracking videos")

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        self.pbar.close()

    def on_video_loop_start(self, engine: "TrackingEngine", video: Any, video_idx: int, index: int):
        n = index
        total = len(engine.video_metadatas)

    def on_video_loop_end(self, engine: "TrackingEngine", video: Any, video_idx: int,
                          detections: "Detections"):
        self.pbar.update()
        self.pbar.refresh()

    def on_task_start(self, engine, task: str, dataloader):
        desc = task.replace("_", " ").capitalize()
        self.task_pbars[task]: tqdm = tqdm(total=len(dataloader), desc=desc,
                                           leave=False, position=1)

    def on_task_step_end(self, engine, task, batch, detections):
        self.task_pbars[task].update()

    def on_task_end(self, engine, task: str, detections: Detections):
        self.task_pbars[task].close()
