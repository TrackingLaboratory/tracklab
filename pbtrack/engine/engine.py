from functools import partial
from typing import Dict, TYPE_CHECKING

import torch
import pbtrack
from lightning.fabric import Fabric

from abc import abstractmethod, ABC

if TYPE_CHECKING:
    from pbtrack.callbacks import Callback

from pbtrack.datastruct import Detections, TrackerState


# noinspection PyCallingNonCallable
class TrackingEngine(ABC):
    """ Manages the full tracking pipeline.

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
                 detect_multi_model: "pbtrack.Detector",
                 detect_single_model: "pbtrack.Detector",
                 reid_model: "pbtrack.ReIdentifier",
                 track_model: "pbtrack.Tracker",
                 tracker_state: "TrackerState",
                 num_workers: int,
                 callbacks: "Dict[Callback]" = None,
                 ):
        # super().__init__()
        callbacks = list(callbacks.values()) if callbacks is not None else []
        callbacks = [tracker_state] + callbacks

        self.fabric = Fabric(callbacks=callbacks)
        self.callback = partial(self.fabric.call, engine=self)
        self.num_workers = num_workers
        self.tracker_state = tracker_state
        self.img_metadatas = tracker_state.gt.image_metadatas
        self.video_metadatas = tracker_state.gt.video_metadatas

        self.models = {"detect_multi": detect_multi_model,
                       "detect_single": detect_single_model, "reid": reid_model,
                       "track": track_model}
        self.datapipes = {}
        self.dataloaders = {}
        for model_name, model in self.models.items():
            self.datapipes[model_name] = model.datapipe
            self.dataloaders[model_name] = model.dataloader(engine=self)

    def track_dataset(self):
        """Run tracking on complete dataset."""
        self.callback("on_dataset_track_start")
        for i, (video_idx, video) in enumerate(self.video_metadatas.iterrows()):
            self.callback("on_video_loop_start", video=video, video_idx=video_idx, index=i)
            detections = self.video_loop(video, video_idx)
            self.callback("on_video_loop_end",
                          video=video,
                          video_idx=video_idx,
                          detections=detections)
        self.callback("on_dataset_track_end")

    @abstractmethod
    def video_loop(self, video, video_id) -> "Detections":
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

    def default_step(self, batch, task, detections: "Detections" = None, **kwargs):
        self.callback(f"on_task_step_start", task=task, batch=batch)
        idxs, batch = batch
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        if detections is None:
            batch_metadatas = self.img_metadatas.loc[idxs]
            detections = Detections(self.models[task].process(batch, batch_metadatas))
        else:
            batch_detections = detections.loc[idxs]
            batch_metadatas = self.img_metadatas.loc[batch_detections.image_id]
            detections = Detections(
                self.models[task].process(batch=batch,
                                          detections=batch_detections,
                                          metadatas=batch_metadatas, **kwargs))
        self.callback(f"on_task_step_end", task=task, batch=batch, detections=detections)
        return detections

    def detect_multi_step(self, batch):
        detections = self.default_step(batch, "detect_multi")
        return detections

    def detect_single_step(self, batch, detections):
        output_detections = self.default_step(batch, "detect_single", detections)
        return output_detections

    def reid_step(self, batch, detections):
        output_detections = self.default_step(batch, "reid", detections)
        return output_detections

    def track_step(self, batch, detections, image):
        output_detections = self.default_step(batch, "track", detections, image=image)
        return output_detections

