import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from pbtrack.utils.coordinates import clip_bbox_ltrb_to_img_dim
from pbtrack.utils.images import cv2_load_image, overlay_heatmap

import matplotlib.cm as cm

cmap = (
    (255 * cm.colors.to_rgba_array(cm.get_cmap("tab10")(range(10))))[:, :3]
    .astype(int)
    .tolist()
)


# TODO show person's squeleton
class VisEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = Path("visualization")
        self.processed_video_counter = 0

    def run(self, tracker_state, video_id):
        video_save_dir = self.save_dir / str(video_id)
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
            detections_gt = tracker_state.gt.detections[
                tracker_state.gt.detections.image_id == image_metadata.id
            ]
            self._process_video_frame(
                image_metadata, detections, detections_gt, video_save_dir, video_id
            )
        if self.cfg.save_videos:
            self.video_writer.release()
            delattr(self, "video_writer")
        self.processed_video_counter += 1

    def _process_video_frame(
        self, image_metadata, detections, detections_gt, video_save_dir, video_id
    ):
        patch = cv2_load_image(image_metadata.file_path)

        # bboxes
        if self.cfg.bbox.show_preds:
            patch = self._plot_bbox(detections, patch, preds=True)
        if self.cfg.bbox.show_gt:
            patch = self._plot_bbox(detections_gt, patch, preds=False)
        # keypoints
        if self.cfg.keypoints.show_preds:
            patch = self._plot_keypoints(detections, patch, preds=True)
        if self.cfg.keypoints.show_gt:
            patch = self._plot_keypoints(detections_gt, patch, preds=False)
        # bpbreid
        if self.cfg.bpbreid.show_heatmaps:
            patch = self._plot_bpbreid_heatmaps(detections, patch)

        patch = self._final_patch(patch)
        if self.cfg.save_images:
            filepath = video_save_dir / "images" / Path(image_metadata.file_path).name
            filepath.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(filepath), patch)
        if self.cfg.save_videos:
            self._update_video(patch, video_save_dir, video_id)

    def _plot_bbox(self, detections, patch, preds):
        for _, detection in detections.iterrows():
            if self.cfg.bbox.show_only_tracked and not detection.track_id:
                continue
            if not pd.isna(detection.track_id):
                color = (
                    cmap[detection.track_id % len(cmap)]
                    if preds
                    else self.cfg.bbox.color_gt
                )
            else:
                color = self.cfg.bbox.color_preds if preds else self.cfg.bbox.color_gt
            bbox = detection.bbox_ltrb
            bbox = clip_bbox_ltrb_to_img_dim(
                bbox, patch.shape[1], patch.shape[0]
            ).astype(int)
            cv2.rectangle(
                patch,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=color,
                thickness=self.cfg.bbox.thickness,
                lineType=cv2.LINE_AA,
            )
            if preds and self.cfg.bbox.print_id:
                cv2.putText(
                    patch,
                    f"ID: {detection.track_id}",
                    (bbox[0] + 2, bbox[1] - 5),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=self.cfg.bbox.font_scale,
                    color=self.cfg.bbox.font_color,
                    thickness=self.cfg.bbox.font_thickness,
                    lineType=cv2.LINE_AA,
                )
            if preds and self.cfg.bbox.print_confidence:
                cv2.putText(
                    patch,
                    f"Conf: {np.mean(detection.keypoints_xyc[:,2]):.2}",
                    (bbox[0] + 2, bbox[3] - 5),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=self.cfg.bbox.font_scale,
                    color=self.cfg.bbox.font_color,
                    thickness=self.cfg.bbox.font_thickness,
                    lineType=cv2.LINE_AA,
                )
        return patch

    def _plot_keypoints(self, detections, patch, preds):
        color = self.cfg.keypoints.color_preds if preds else self.cfg.keypoints.color_gt
        all_keypoints = detections.keypoints_xyc
        for keypoints in all_keypoints:
            for kp in keypoints:
                cv2.circle(
                    patch,
                    (int(kp[0]), int(kp[1])),
                    radius=self.cfg.keypoints.radius,
                    color=color,
                    thickness=self.cfg.keypoints.thickness,
                    lineType=cv2.LINE_AA,
                )
                if preds and self.cfg.keypoints.print_confidence:
                    cv2.putText(
                        patch,
                        f"{kp[2]:.2}",
                        (int(kp[0]) + 2, int(kp[1]) - 5),
                        fontScale=self.cfg.keypoints.font_scale,
                        color=self.cfg.keypoints.font_color,
                        thickness=self.cfg.keypoints.font_thickness,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        lineType=cv2.LINE_AA,
                    )
        return patch

    def _plot_bpbreid_heatmaps(self, detections, patch):
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
