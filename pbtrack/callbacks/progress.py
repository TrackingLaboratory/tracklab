import pandas as pd

from typing import Any, Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from pbtrack.callbacks import Callback
from pbtrack.engine import TrackingEngine


class Progressbar(Callback):
    def __init__(self):
        self.pbar: Optional[tqdm] = None
        self.task_pbars = {}

    def on_dataset_track_start(self, engine: TrackingEngine):
        total = len(engine.video_metadatas)
        self.pbar = tqdm(total=total, desc="Tracking videos")

    def on_dataset_track_end(self, engine: TrackingEngine):
        self.pbar.close()

    def on_video_loop_start(
        self, engine: TrackingEngine, video: Any, video_idx: int, index: int
    ):
        n = index
        total = len(engine.video_metadatas)
        self.video_id = video_idx

    def on_video_loop_end(
        self,
        engine: TrackingEngine,
        video: Any,
        video_idx: int,
        detections: pd.DataFrame,
    ):
        self.pbar.update()
        self.pbar.refresh()

    def on_task_start(self, engine: TrackingEngine, task: str, dataloader: DataLoader):
        desc = task.replace("_", " ").capitalize()
        if hasattr(engine.models[task], "process_video"):
            length = len(engine.img_metadatas[engine.img_metadatas.video_id == self.video_id])
        else:
            length = len(dataloader)
        self.task_pbars[task]: tqdm = tqdm(
            total=length, desc=desc, leave=False, position=1
        )

    def on_task_step_end(
        self, engine: TrackingEngine, task: str, batch: Any, detections: pd.DataFrame
    ):
        self.task_pbars[task].update()

    def on_task_end(self, engine: TrackingEngine, task: str, detections: pd.DataFrame):
        self.task_pbars[task].close()
