import pandas as pd
from pbtrack.datastruct import Detections

from pbtrack.engine import TrackingEngine
from pbtrack.utils.images import cv2_load_image


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, video, video_id):
        with self.tracker_state(video_id) as tracker_state:
            self.models["track"].reset()
            imgs_meta = self.img_metadatas[self.img_metadatas.video_id == video_id]
            detections = tracker_state.load()
            loaded = detections
            if tracker_state.do_detect_multi or not loaded:
                self.datapipes["detect_multi"].update(imgs_meta)
                self.callback("on_task_start", task="detect_multi",
                              dataloader=self.dataloaders["detect_multi"])
                detections_list = []
                for batch in self.dataloaders["detect_multi"]:
                    detections_list.append(self.detect_multi_step(batch))
                detections = pd.concat(detections_list)
                self.callback("on_task_end", task="detect_multi", detections=detections)
            if detections.empty:
                return detections

            if self.models["detect_single"] and (
                    tracker_state.do_detect_single or not loaded):
                self.datapipes["detect_single"].update(imgs_meta, detections)
                self.callback("on_task_start", task="detect_single",
                              dataloader=self.dataloaders["detect_single"])
                detections_list = []
                for batch in self.dataloaders["detect_single"]:
                    detections_list.append(self.detect_single_step(batch, detections))
                detections = pd.concat(detections_list)
                self.callback("on_task_end", task="detect_single",
                              detections=detections)

                if tracker_state.do_reid or not loaded:
                    self.datapipes["reid"].update(imgs_meta, detections)
                    self.callback("on_task_start", task="reid",
                                  dataloader=self.dataloaders["reid"])
                    detections_list = []
                    for batch in self.dataloaders["reid"]:
                        detections_list.append(self.reid_step(batch, detections))
                    detections = pd.concat(detections_list)
                    self.callback("on_task_end", task="reid",
                                  detections=detections)

                if tracker_state.do_tracking or not loaded:
                    self.callback("on_task_start", task="track",
                                  dataloader=[])
                    detections = self.models["track"].process_video(detections,
                                                                    imgs_meta,
                                                                    self)
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
                    self.callback("on_task_end", task="track",
                                  detections=detections)

        return detections

    def loop_step(self, step, loaded, tracker_state, imgs_meta):
        if tracker_state.do["step"] or not loaded:
            self.callback(f"{step}_loop_start")
            self.datapipes[step].update(imgs_meta)
            detections_list = []
            for batch, detections in self.dataloaders[step]:
                detections_list.append(getattr(self, f"{step}_step")(batch, detections))

            self.callback(f"{step}_loop_end")
            return pd.concat(detections_list)
