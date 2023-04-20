import pandas as pd

from pbtrack.engine import TrackingEngine


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, video, video_id):
        with self.tracker_state(video_id) as tracker_state:
            self.models["track"].reset()
            imgs_meta = self.img_metadatas[self.img_metadatas.video_id == video_id]
            detections = tracker_state.load()
            loaded = detections
            if tracker_state.do_detect_multi or not loaded:
                self.datapipes["detect_multi"].update(imgs_meta)
                self.callback(
                    "on_task_start",
                    task="detect_multi",
                    dataloader=self.dataloaders["detect_multi"],
                )
                detections = pd.DataFrame()
                for batch in self.dataloaders["detect_multi"]:
                    detections = self.detect_multi_step(batch, detections)
                self.callback("on_task_end", task="detect_multi", detections=detections)

            if self.models["detect_single"] and (
                tracker_state.do_detect_single or not loaded
            ):
                self.datapipes["detect_single"].update(imgs_meta, detections)
                self.callback(
                    "on_task_start",
                    task="detect_single",
                    dataloader=self.dataloaders["detect_single"],
                )
                for batch in self.dataloaders["detect_single"]:
                    detections = self.detect_single_step(batch, detections)
                self.callback(
                    "on_task_end", task="detect_single", detections=detections
                )

            if tracker_state.do_reid or not loaded:
                self.datapipes["reid"].update(imgs_meta, detections)
                self.callback(
                    "on_task_start",
                    task="reid",
                    dataloader=self.dataloaders["reid"],
                )
                for batch in self.dataloaders["reid"]:
                    detections = self.reid_step(batch, detections)
                self.callback("on_task_end", task="reid", detections=detections)

            if tracker_state.do_tracking or not loaded:
                self.callback("on_task_start", task="track", dataloader=[])
                detections = self.models["track"].process_video(detections, imgs_meta, self)
                # track_detections = []
                # for image_id in imgs_meta.index:
                #     image = cv2_load_image(imgs_meta.loc[image_id].file_path)
                #     self.models["track"].prepare_next_frame(image)
                #     image_detections = detections[detections.image_id == image_id]
                #     if len(image_detections) != 0:
                #         self.datapipes["track"].update(imgs_meta, image_detections)
                #         for batch in self.dataloaders["track"]:
                #             track_detections.append(
                #                 self.track_step(batch, detections, image)
                #             )
                #
                # if len(track_detections) > 0:
                #     detections = pd.concat(track_detections)
                # else:
                #     detections = Detections()
                self.callback("on_task_end", task="track", detections=detections)

        return detections
