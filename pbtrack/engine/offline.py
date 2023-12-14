import logging

from pbtrack.engine import TrackingEngine

log = logging.getLogger(__name__)


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, tracker_state, video, video_id):
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()

        imgs_meta = self.img_metadatas[self.img_metadatas.video_id == video_id]
        detections = tracker_state.load()
        start = tracker_state.load_index
        log.info(
            f"Inference will be composed of the following steps: {', '.join(x for x in self.module_names[start:])}"
        )
        model_names = self.module_names[start:]
        # print('in offline.py, model_names: ', model_names)
        for model_name in model_names:
            print('in offline.py, model_name: ', model_name)
            # print('in offline.py, self.datapipes: ', self.datapipes)
            # print('in offline.py, len(detections): ', len(detections))
            self.datapipes[model_name].update(imgs_meta, detections)
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
        # print('in offline.py, detections: ', detections)
        print('in offline.py, detections saved to detections.pkl')
        detections.to_pickle('detections.pkl')
        return detections
