from functools import partial
from typing import Dict, TYPE_CHECKING, Any, List

import numpy as np
import pandas as pd
import torch
import pbtrack
from lightning.fabric import Fabric

from abc import abstractmethod, ABC

from pbtrack.pipeline import Pipeline, Module

if TYPE_CHECKING:
    from pbtrack.callbacks import Callback

from pbtrack.datastruct import TrackerState


def merge_dataframes(main_df, appended_piece):
    # Convert appended_piece to a DataFrame if it's not already
    if isinstance(appended_piece, pd.Series):
        appended_piece = pd.DataFrame(appended_piece).T
    elif isinstance(appended_piece, list):  # list of Series or DataFrames
        if len(appended_piece) > 0:
            appended_piece = pd.concat(
                [s.to_frame().T if type(s) is pd.Series else s for s in appended_piece]
            )
        else:
            appended_piece = pd.DataFrame()

    # Append the columns of the df
    new_columns = appended_piece.columns.difference(main_df.columns)
    for column in new_columns:
        main_df[column] = np.nan

    # Append the rows of the df
    new_index = set(appended_piece.index).difference(main_df.index)
    for index in new_index:
        main_df.loc[index] = np.nan

    # Update all the values (appended_piece overrides)
    main_df.update(appended_piece)
    return main_df


class TrackingEngine(ABC):
    """Manages the full tracking pipeline.

    After initializing the ``TrackingEngine``, users should call ``track_dataset``
    which will track each video in turn. The call stack looks like :

    track_dataset
    |
    video_step
    |
    -> detect_multi_step -> detect_single_step -> reid_step -> track_step

    Implementors of ``TrackingEngine`` will need to *at least* implement ``video_step``.
    for example, an online engine will simply call each step in turn for every image in
    a video. An offline engine might instead call each step for all the images before
    doing the next step in the pipeline.

    You should take care to implement the different callback hooks, by calling::
        self.fabric.call("a_callback_function", *args, **kwargs)

    Args:
        detect_multi_model: The bbox/pose detection model
        detect_single_model: The pose detection model
        reid_model: Reid model
        track_model: tracking model
        tracker_state: contains inputs and outputs
        callbacks: called at different steps
        num_workers: number of workers for preprocessing
    """

    def __init__(self,
                 modules: Pipeline,
                 tracker_state: "TrackerState",
                 num_workers: int,
                 callbacks: "Dict[Callback]" = None,
                 ):
        # super().__init__()
        self.module_names = [module.name for module in modules]
        callbacks = list(callbacks.values()) if callbacks is not None else []
        callbacks = [tracker_state] + callbacks

        self.fabric = Fabric(callbacks=callbacks)
        self.callback = partial(self.fabric.call, engine=self)
        self.num_workers = num_workers
        self.tracker_state = tracker_state
        self.img_metadatas = tracker_state.gt.image_metadatas
        self.video_metadatas = tracker_state.gt.video_metadatas
        self.models = {model.name: model for model in modules}
        self.datapipes = {}
        self.dataloaders = {}
        for model_name, model in self.models.items():
            self.datapipes[model_name] = model.datapipe
            self.dataloaders[model_name] = model.dataloader(engine=self)

    def track_dataset(self):
        """Run tracking on complete dataset."""
        self.callback("on_dataset_track_start")
        for i, (video_idx, video) in enumerate(self.video_metadatas.iterrows()):
            self.callback(
                "on_video_loop_start", video=video, video_idx=video_idx, index=i
            )
            detections = self.video_loop(video, video_idx)
            self.callback(
                "on_video_loop_end",
                video=video,
                video_idx=video_idx,
                detections=detections,
            )
        self.callback("on_dataset_track_end")

    @abstractmethod
    def video_loop(self, video, video_id) -> pd.DataFrame:
        """Run tracking on one video.

        The pipeline for each video looks like :

        detect_multi -> (detect_single) -> reid -> track

        Args:
            video: ...
            video_id: ...

        Returns:
            detections: a dataframe of all detections
        """
        pass

    def default_step(self, batch: Any, task: str, detections: pd.DataFrame, **kwargs):
        self.callback(f"on_task_step_start", task=task, batch=batch)
        idxs, batch = batch
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        if task == "multi_detector":
            batch_metadatas = self.img_metadatas.loc[idxs]
            batch_detections = self.models[task].process(batch, batch_metadatas)
        else:
            batch_detections = detections.loc[idxs]
            batch_detections = self.models[task].process(
                batch=batch,
                detections=batch_detections,
                **kwargs,
            )
        detections = merge_dataframes(detections, batch_detections)
        self.callback(
            f"on_task_step_end", task=task, batch=batch, detections=detections
        )
        return detections
