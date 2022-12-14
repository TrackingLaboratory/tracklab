import os
import cv2
import numpy as np

from pbtrack.utils.images import cv2_load_image


class VisEngine:
    def __init__(self, vis_cfg, save_dir):
        self.cfg = vis_cfg
        self.save_dir = save_dir

        self.save_img = self.cfg["images"]["save"]
        self.save_vid = self.cfg["video"]["save"]

        if self.save_img:
            self.save_image_dir = os.path.join(self.save_dir, "images")
            os.makedirs(self.save_image_dir, exist_ok=True)

        if self.save_vid:
            self.save_video_name = os.path.join(self.save_dir, "results.mp4")

    def run(self, tracker_state):
        for video in tracker_state.gt.video_metadatas.itertuples():

            image_metadatas = tracker_state.gt.image_metadatas[
                tracker_state.gt.image_metadatas.video_id == video.id
            ]  # FIXME sort?
            # for image_metadata in image_metadatas.itertuples():  # FIXME image_metadata is of type Panda, not ImageMetadata
            for image_id in image_metadatas.id:
                image_metadata = image_metadatas.loc[image_id]
                detections = tracker_state.predictions[
                    tracker_state.predictions.image_id == image_metadata.id
                ]
                self._process_video_frame(image_metadata, detections)
            if self.save_vid:
                self.video.release()

    def _process_video_frame(self, image_metadata, detections):
        # loop on GT
        image = cv2_load_image(image_metadata.file_path)
        # image = self._process_img(image)

        # data = file_name, width, height,

        if self.cfg["detection"]["bbox"]["show"]:
            image = self._plot_bbox(detections, image)
        if self.cfg["detection"]["pose"]["show"]:
            image = self._plot_pose(detections, image)
        if self.cfg["tracking"]["show"]:
            image = self._plot_track(detections, image)

        image = self._final_patch(image)
        if self.save_img:
            filepath = os.path.join(
                self.save_image_dir, os.path.basename(image_metadata.file_path)
            )
            assert cv2.imwrite(filepath, image)
        # save video
        if self.save_vid:
            self._update_video(image)

    def _plot_bbox(self, detections, patch):
        bboxes = detections.bbox_ltrb
        for bbox in bboxes:
            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(
                patch,
                p1,
                p2,
                color=self.cfg["detection"]["bbox"]["color"],
                thickness=self.cfg["detection"]["bbox"]["thickness"],
            )
            if self.cfg["detection"]["bbox"]["print_conf"]:
                p = (int(bbox[0]) + 1, int(bbox[1]) - 2)
                cv2.putText(
                    patch,
                    f" {bbox[4]:.2}",
                    p,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=0.75,
                    color=self.cfg["detection"]["bbox"]["color"],
                    thickness=1,
                )
        return patch

    def _plot_pose(self, detections, patch):
        poses = detections.keypoints_xyc
        for pose in poses:
            for kp in pose:
                p = (int(kp[0]), int(kp[1]))
                cv2.circle(
                    patch,
                    p,
                    radius=self.cfg["detection"]["pose"]["radius"],
                    color=self.cfg["detection"]["pose"]["color"],
                    thickness=self.cfg["detection"]["pose"]["thickness"],
                )
                if self.cfg["detection"]["pose"]["print_conf"]:
                    p = (int(kp[0]) + 1, int(kp[1]) - 2)
                    cv2.putText(
                        patch,
                        f"{kp[2]:.2}",
                        p,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=0.75,
                        color=self.cfg["detection"]["pose"]["color"],
                        thickness=1,
                    )
        return patch

    def _plot_track(self, detections, patch):
        bboxes = detections.bbox_ltrb
        ids = detections[["person_id"]].values
        for bbox, id in zip(bboxes, ids):
            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(
                patch,
                p1,
                p2,
                color=self.cfg["tracking"]["color"],
                thickness=self.cfg["tracking"]["thickness"],
            )
            txt = ""
            if self.cfg["tracking"]["print_id"]:
                txt += f" ID: {id}"
            if self.cfg["tracking"]["print_conf"]:
                txt += f" - conf: {bbox[4]:.2}"
            p = (int(bbox[0]) + 1, int(bbox[1]) - 2)
            cv2.putText(
                patch,
                txt,
                p,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale=0.75,
                color=self.cfg["tracking"]["color"],
                thickness=1,
            )
        return patch

    def _update_video(self, patch):
        if not hasattr(self, "video"):
            self.video = cv2.VideoWriter(
                self.save_video_name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                int(self.cfg["video"]["fps"]),
                (int(self.cfg["video"]["height"]), int(self.cfg["video"]["width"])),
            )
        self.video.write(patch)

    def _process_img(self, img):
        img = 255.0 * img
        img = img.transpose(1, 2, 0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img

    def _final_patch(self, patch):
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        patch = patch.astype(np.uint8)
        return patch
