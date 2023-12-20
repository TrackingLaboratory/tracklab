import logging

from pbtrack.engine import TrackingEngine
from pbtrack.utils.cv2 import cv2_load_image

log = logging.getLogger(__name__)


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, tracker_state, video, video_id):
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()

        imgs_meta = self.img_metadatas[self.img_metadatas.video_id == video_id]
        images = {idx: cv2_load_image(fn) for idx, fn in imgs_meta["file_path"].items()}
        detections = tracker_state.load()
        start = tracker_state.load_index
        model_names = self.module_names[start:]
        for model_name in model_names:
            self.datapipes[model_name].update(images, imgs_meta, detections)
            self.callback(
                "on_task_start",
                task=model_name,
                dataloader=self.dataloaders[model_name],
            )
            if hasattr(self.models[model_name], "process_video"):
                detections = self.models[model_name].process_video(
                    detections, imgs_meta, self
                )
            else:
                for batch in self.dataloaders[model_name]:
                    detections = self.default_step(batch, model_name, detections)
            self.callback("on_task_end", task=model_name, detections=detections)
            if detections.empty:
                return detections

        return detections
