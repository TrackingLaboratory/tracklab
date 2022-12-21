import cv2
import numpy as np

from pathlib import Path

from pbtrack.utils.coordinates import clip_bbox_ltrb_to_img_dim
from pbtrack.utils.images import cv2_load_image, overlay_heatmap


# TODO add automatic colors by ID
# TODO add ID print next to bbox
# TODO show person's squeleton
# TODO add possibility to visualize the groundtruths
class VisEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = Path("visualization")
        self.processed_video_counter = 0

    def run(self, tracker_state, video_id):
        video_save_dir = self.save_dir / tracker_state.gt.split / str(video_id)
        if self.processed_video_counter >= self.cfg.process_n_videos != -1:
            return
        image_metadatas = tracker_state.gt.image_metadatas[
            tracker_state.gt.image_metadatas.video_id == video_id
        ]  # FIXME sort?
        for i, image_id in enumerate(image_metadatas.id):
            if i > self.cfg.process_n_frames_by_video != -1:
                break
            image_metadata = image_metadatas.loc[image_id]
            detections = tracker_state.predictions[
                tracker_state.predictions.image_id == image_metadata.id
            ]
            self._process_video_frame(
                image_metadata, detections, video_save_dir, video_id
            )
        if self.cfg.save_videos:
            self.video_writer.release()
            delattr(self, "video_writer")
        self.processed_video_counter += 1

    def _process_video_frame(
        self, image_metadata, detections, video_save_dir, video_id
    ):
        image = cv2_load_image(image_metadata.file_path)

        if self.cfg.bbox.show:
            image = self._plot_bbox(detections, image)
        if self.cfg.keypoints.show:
            image = self._plot_keypoints(detections, image)
        if self.cfg.heatmaps.show:
            image = self._show_heatmaps(detections, image)

        image = self._final_patch(image)
        if self.cfg.save_images:
            filepath = video_save_dir / "images" / Path(image_metadata.file_path).name
            filepath.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(filepath), image)
        if self.cfg.save_videos:
            self._update_video(image, video_save_dir, video_id)

    def _plot_bbox(self, detections, patch):
        for i, detection in detections.iterrows():
            bbox = detection.bbox_ltrb
            bbox = clip_bbox_ltrb_to_img_dim(
                bbox, patch.shape[1], patch.shape[0]
            ).astype(int)
            p1, p2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
            cv2.rectangle(
                patch,
                p1,
                p2,
                color=self.cfg.bbox.color,
                thickness=self.cfg.bbox.thickness,
            )
            # TODO add ID
            if self.cfg.bbox.print_confidence:
                p = (bbox[0] + 1, bbox[1] - 2)
                cv2.putText(
                    patch,
                    f"{np.mean(detection.keypoints_xyc[:,2]):.2}",
                    p,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=self.cfg.bbox.font_scale,
                    color=self.cfg.bbox.font_color,
                    thickness=self.cfg.bbox.font_thickness,
                )
        return patch

    def _plot_keypoints(self, detections, patch):
        all_keypoints = detections.keypoints_xyc
        for keypoints in all_keypoints:
            for kp in keypoints:
                p = (int(kp[0]), int(kp[1]))
                cv2.circle(
                    patch,
                    p,
                    radius=self.cfg.keypoints.radius,
                    color=self.cfg.keypoints.color,
                    thickness=self.cfg.keypoints.thickness,
                )
                if self.cfg.keypoints.print_confidence:
                    p = (int(kp[0] + 1), int(kp[1] - 2))
                    cv2.putText(
                        patch,
                        f"{kp[2]:.2}",
                        p,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=self.cfg.keypoints.font_scale,
                        color=self.cfg.keypoints.font_color,
                        thickness=self.cfg.keypoints.font_thickness,
                    )
        return patch

    def _show_heatmaps(self, detections, patch):
        for i, detection in detections.iterrows():
            ltrb = detection.bbox_ltrb
            l, t, r, b = clip_bbox_ltrb_to_img_dim(
                ltrb, patch.shape[1], patch.shape[0]
            ).astype(int)
            img_crop = patch[t:b, l:r]
            body_masks = detection.body_masks
            img_crop_with_mask = overlay_heatmap(img_crop, body_masks[0], rgb=True)
            patch[t:b, l:r] = img_crop_with_mask
        return patch

    def _update_video(self, patch, video_save_dir, video_id):
        if not hasattr(self, "video_writer"):
            filepath = video_save_dir / "video" / f"{video_id}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.cfg.video_fps),
                (patch.shape[1], patch.shape[0]),
            )
        self.video_writer.write(patch)

    def _final_patch(self, patch):
        return cv2.cvtColor(patch, cv2.COLOR_RGB2BGR).astype(np.uint8)
