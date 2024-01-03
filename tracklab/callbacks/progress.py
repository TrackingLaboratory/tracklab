import pandas as pd
import logging

from typing import Any, Optional
from rich.progress import Progress
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracklab.callbacks import Callback
from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)

class Progressbar(Callback):
    def __init__(self):
        self.pbar: Optional[tqdm] = None
        self.task_pbars = {}

    def on_dataset_track_start(self, engine: TrackingEngine):
        total = len(engine.video_metadatas)
        log.info(f"Inference will be composed of the following steps: {', '.join(x for x in engine.module_names)}")
        self.pbar = tqdm(total=total, desc="Tracking videos")

    def on_dataset_track_end(self, engine: TrackingEngine):
        self.pbar.close()

    def on_video_loop_start(
        self, engine: TrackingEngine, video_metadata: pd.Series, video_idx: int, index: int
    ):
        n = index
        total = len(engine.video_metadatas)
        self.video_id = video_idx

    def on_video_loop_end(
        self,
        engine: TrackingEngine,
        video_metadata: pd.Series,
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


class RichProgressbar(Progressbar):
    def __init__(self):
        self.pbar: Optional[Progress] = None
        self.tasks = {}

    def on_dataset_track_start(self, engine: TrackingEngine):
        total = len(engine.video_metadatas)
        self.pbar = Progress()
        self.pbar.start()
        self.tasks["main"] = self.pbar.add_task("[yellow]Tracking videos", total=total)

    def on_dataset_track_end(self, engine: TrackingEngine):
        self.pbar.stop()

    def on_video_loop_start(
        self, engine: TrackingEngine, video_metadata: pd.Series, video_idx: int, index: int
    ):
        self.video_id = video_idx

    def on_video_loop_end(
        self,
        engine: TrackingEngine,
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
    ):
        self.pbar.update(self.tasks["main"], advance=1, refresh=True)

    def on_task_start(self, engine: TrackingEngine, task: str, dataloader: DataLoader):
        desc = task.replace("_", " ").capitalize()
        if hasattr(engine.models[task], "process_video"):
            length = len(engine.img_metadatas[engine.img_metadatas.video_id == self.video_id])
        else:
            length = len(dataloader)
        self.tasks[task] = self.pbar.add_task(desc, total=length)

    def on_task_step_end(
        self, engine: TrackingEngine, task: str, batch: Any, detections: pd.DataFrame
    ):
        self.pbar.update(self.tasks[task], advance=1)

    def on_task_end(self, engine: TrackingEngine, task: str, detections: pd.DataFrame):
        self.pbar.stop_task(self.tasks[task])
        self.pbar.remove_task(self.tasks[task])
