# from pytorch_lightning import Callback as PLCallback
from typing import Any, TYPE_CHECKING

import pandas as pd
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pbtrack.engine import TrackingEngine


class Callback:
    def on_dataset_track_start(self, engine: "TrackingEngine"):
        pass

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        pass

    def on_video_loop_start(
        self, engine: "TrackingEngine", video: Any, video_idx: int, index: int
    ):
        pass

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video: Any,
        video_idx: int,
        detections: pd.DataFrame,
    ):
        pass

    def on_task_start(
        self, engine: "TrackingEngine", task: str, dataloader: DataLoader
    ):
        pass

    def on_task_end(
        self, engine: "TrackingEngine", task: str, detections: pd.DataFrame
    ):
        pass

    def on_task_step_start(self, engine: "TrackingEngine", task: str, batch: Any):
        pass

    def on_task_step_end(
        self, engine: "TrackingEngine", task: str, batch: Any, detections: pd.DataFrame
    ):
        pass
