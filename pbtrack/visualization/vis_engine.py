from pathlib import Path

import cv2
import numpy as np

from pbtrack.utils.images import cv2_load_image, overlay_heatmap


class VisEngine:
    def __init__(self, vis_cfg, save_dir):
        self.cfg = vis_cfg
        self.save_dir = Path(save_dir) / "vis"

        self.save_img = self.cfg["images"]["save"]
        self.save_vid = self.cfg["video"]["save"]

        self.video_counter = 0

    def run(self, tracker_state, video_id):
        video_save_dir = self.save_dir / tracker_state.gt.split / str(video_id)
        if self.video_counter > self.cfg.max_videos != -1:
            return
        image_metadatas = tracker_state.gt.image_metadatas[
            tracker_state.gt.image_metadatas.video_id == video_id
        ]  # FIXME sort?
        for i, image_id in enumerate(image_metadatas.id):
            if i > self.cfg.max_frames != -1:
                break
            image_metadata = image_metadatas.loc[image_id]
            detections = tracker_state.predictions[
                tracker_state.predictions.image_id == image_metadata.id
            ]
            self._process_video_frame(
                image_metadata, detections, video_save_dir, video_id
            )
        if self.save_vid:
            self.video.release()

    def _process_video_frame(
        self, image_metadata, detections, video_save_dir, video_id
    ):
        # loop on GT
        image = cv2_load_image(image_metadata.file_path)
        # image = self._process_img(image)

        # data = file_name, width, height,

        if self.cfg["detection"]["bbox"]["show"]:
            image = self._plot_bbox(detections, image)
        if self.cfg["detection"]["pose"]["show"]:
            image = self._plot_pose(detections, image)
        if self.cfg["detection"]["heatmaps"]["show"]:
            image = self._show_heatmaps(detections, image)
        if self.cfg["tracking"]["show"]:
            image = self._plot_track(detections, image)

        image = self._final_patch(image)
        if self.save_img:
            filepath = video_save_dir / "images" / Path(image_metadata.file_path).name
            filepath.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(filepath), image)
        # save video
        if self.save_vid:
            self._update_video(image, video_save_dir, video_id)

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

    def _show_heatmaps(self, detections, patch):
        for i, detection in detections.iterrows():
            l, t, r, b = detection.bbox_ltrb.astype(np.int)
            body_masks = detection.body_masks
            img_crop = patch[t:b, l:r]
            img_crop_with_mask = overlay_heatmap(img_crop, body_masks[0], rgb=True)
            patch[t:b, l:r] = img_crop_with_mask
        return patch

    def _plot_track(self, detections, patch):
        bboxes = detections.bbox_ltrb
        ids = detections[["track_id"]].values
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

    def _update_video(self, patch, video_save_dir, video_id):
        if not hasattr(self, "video"):
            filepath = video_save_dir / "video" / f"{video_id}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.video = cv2.VideoWriter(
                str(filepath),
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
