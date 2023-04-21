import logging

import pandas as pd

from pbtrack.engine import TrackingEngine

log = logging.getLogger(__name__)


class OfflineTrackingEngine(TrackingEngine):

    def video_loop(self, video, video_id):
        with self.tracker_state(video_id) as tracker_state:
            for name, model in self.models.items():
                if hasattr(model, "reset"):
                    model.reset()

            imgs_meta = self.img_metadatas[self.img_metadatas.video_id == video_id]
            detections = tracker_state.load()
            start = tracker_state.load_index
            log.info(f"Starting from {self.module_names[start]}")
            model_names = self.module_names[start:]
            for model_name in model_names:
                self.datapipes[model_name].update(imgs_meta, detections)
                self.callback("on_task_start", task=model_name,
                              dataloader=self.dataloaders[model_name])
                if hasattr(self.models[model_name], "process_video"):
                    detections = self.models[model_name].process_video(detections,
                                                                       imgs_meta,
                                                                       self)
                else:
                    for batch in self.dataloaders[model_name]:
                        detections = self.default_step(batch, model_name, detections)
                self.callback("on_task_end", task=model_name, detections=detections)
                if detections.empty:
                    return detections

        return detections