import logging

from tracklab.engine import TrackingEngine
from tracklab.engine.engine import merge_dataframes
from tracklab.utils.cv2 import cv2_load_image

log = logging.getLogger(__name__)


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, tracker_state, video, video_id):
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()

        detections, image_pred = tracker_state.load()
        if len(self.module_names) == 0:
            return detections, image_pred
        image_filepaths = {idx: fn for idx, fn in image_pred["file_path"].items()}
        model_names = self.module_names
        for model_name in model_names:
            if self.models[model_name].level == "video":
                batch_detections = self.models[model_name].process(detections, image_pred)
                if isinstance(batch_detections, tuple):
                    batch_detections, batch_metadata = batch_detections
                    image_pred = merge_dataframes(image_pred, batch_metadata)
                detections = merge_dataframes(detections, batch_detections)
                continue
            self.datapipes[model_name].update(image_filepaths, image_pred, detections)
            self.callback(
                "on_module_start",
                task=model_name,
                dataloader=self.dataloaders[model_name],
            )
            for batch in self.dataloaders[model_name]:
                detections, image_pred = self.default_step(batch, model_name, detections, image_pred)
            self.callback("on_module_end", task=model_name, detections=detections)
            if detections.empty:
                return detections, image_pred
        return detections, image_pred
