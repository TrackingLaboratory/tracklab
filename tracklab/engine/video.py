import platform
from functools import partial
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from lightning import Fabric

from tracklab.engine import TrackingEngine
from tracklab.engine.engine import merge_dataframes
from tracklab.pipeline import Pipeline

import logging

log = logging.getLogger(__name__)


class VideoOnlineTrackingEngine:
    def __init__(
        self,
        modules: Pipeline,
        filename: str,
        target_fps: int,
        tracker_state,
        num_workers: int,
        callbacks: "Dict[Callback]" = None,
    ):
        # super().__init__()
        self.module_names = [module.name for module in modules]
        callbacks = list(callbacks.values()) if callbacks is not None else []

        self.fabric = Fabric(callbacks=callbacks)
        self.callback = partial(self.fabric.call, engine=self)
        self.num_workers = num_workers
        self.video_filename = filename
        self.target_fps = target_fps
        self.tracker_state = tracker_state
        self.img_metadatas = tracker_state.image_metadatas
        self.video_metadatas = tracker_state.video_metadatas
        self.models = {model.name: model for model in modules}
        self.datapipes = {}
        self.dataloaders = {}
        for model_name, model in self.models.items():
            self.datapipes[model_name] = getattr(model, "datapipe", None)
            self.dataloaders[model_name] = getattr(model, "dataloader", lambda **kwargs: ...)(engine=self)

    def track_dataset(self):
        """Run tracking on complete dataset."""
        self.callback("on_dataset_track_start")
        self.callback(
            "on_video_loop_start",
            video_metadata=pd.Series(name=self.video_filename),
            video_idx=0,
            index=0,
        )
        detections = self.video_loop()
        self.callback(
            "on_video_loop_end",
            video_metadata=pd.Series(name=self.video_filename),
            video_idx=0,
            detections=detections,
        )
        self.callback("on_dataset_track_end")

    def video_loop(self):
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()
        video_filename = int(self.video_filename) if str(self.video_filename).isnumeric() else str(self.video_filename)
        video_cap = cv2.VideoCapture(video_filename)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_modulo = fps // self.target_fps
        assert video_cap.isOpened(), f"Error opening video stream or file {video_filename}"
        if platform.system() == "Linux":
            cv2.namedWindow(str(self.video_filename), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # cv2.resizeWindow(str(self.video_filename))

        model_names = self.module_names
        # print('in offline.py, model_names: ', model_names)
        frame_idx = -1
        detections = pd.DataFrame()
        while video_cap.isOpened():
            frame_idx += 1
            ret, frame = video_cap.read()
            if frame_idx % frame_modulo != 0:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            metadata = pd.Series({"id": frame_idx, "frame": frame_idx,
                                  "video_id": video_filename}, name=frame_idx)
            self.callback("on_image_loop_start",
                          image_metadata=metadata, image_idx=frame_idx, index=frame_idx)
            for model_name in model_names:
                model = self.models[model_name]
                if len(detections) > 0:
                    dets = detections[detections.image_id == frame_idx]
                else:
                    dets = pd.DataFrame()
                if model.level == "video":
                    raise "Video-level not supported for online video tracking"
                elif model.level == "image":
                    batch = model.preprocess(image=image, detections=dets, metadata=metadata)
                    batch = type(model).collate_fn([(frame_idx, batch)])
                    detections = self.default_step(batch, model_name, detections, metadata)
                elif model.level == "detection":
                    for idx, detection in dets.iterrows():
                        batch = model.preprocess(image=image, detection=detection, metadata=metadata)
                        batch = type(model).collate_fn([(detection.name, batch)])
                        detections = self.default_step(batch, model_name, detections, metadata)
            self.callback("on_image_loop_end",
                          image_metadata=metadata, image=image,
                          image_idx=frame_idx, detections=detections)

        return detections

    def default_step(self, batch: Any, task: str, detections: pd.DataFrame, metadata, **kwargs):
        model = self.models[task]
        self.callback(f"on_module_step_start", task=task, batch=batch)
        idxs, batch = batch
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        if model.level == "image":
            log.info(f"step : {idxs}")
            batch_metadatas = pd.DataFrame([metadata])
            if len(detections) > 0:
                batch_input_detections = detections.loc[
                    np.isin(detections.image_id, batch_metadatas.index)
                ]
            else:
                batch_input_detections = detections
            batch_detections = self.models[task].process(
                batch,
                batch_input_detections,
                batch_metadatas)
        else:
            batch_detections = detections.loc[idxs]
            batch_detections = self.models[task].process(
                batch=batch,
                detections=batch_detections,
                metadatas=None,
                **kwargs,
            )
        detections = merge_dataframes(detections, batch_detections)
        self.callback(
            f"on_module_step_end", task=task, batch=batch, detections=detections
        )
        return detections

