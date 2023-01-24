import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from pbtrack.utils.coordinates import clip_bbox_ltrb_to_img_dim
from pbtrack.utils.images import cv2_load_image, overlay_heatmap

cmap = [
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 212],
    [0, 128, 128],
    [220, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 215, 180],
    [0, 0, 128],
    [128, 128, 128],
]

human_skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]


class VisualisationEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = Path("visualization")
        self.processed_video_counter = 0

    def run(self, tracker_state, video_id):
        # check for process max video
        if self.processed_video_counter >= self.cfg.process_n_videos != -1:
            return

        image_metadatas = tracker_state.gt.image_metadatas[
            tracker_state.gt.image_metadatas.video_id == video_id
        ]
        for i, image_id in enumerate(image_metadatas.id):
            # check for process max frame per video
            if i >= self.cfg.process_n_frames_by_video != -1:
                break
            # retrieve results
            image_metadata = image_metadatas.loc[image_id]
            predictions = tracker_state.predictions[
                tracker_state.predictions.image_id == image_metadata.id
            ]
            if tracker_state.gt.detections is not None:
                ground_truths = tracker_state.gt.detections[
                    tracker_state.gt.detections.image_id == image_metadata.id
                ]
            else:
                ground_truths = None
            # process the detections
            self._process_frame(image_metadata, predictions, ground_truths, video_id)
        # save the final video
        if self.cfg.save_videos:
            self.video_writer.release()
            delattr(self, "video_writer")
        self.processed_video_counter += 1

    def _process_frame(self, image_metadata, predictions, ground_truths, video_id):
        # load image
        patch = cv2_load_image(image_metadata.file_path)
        # draw predictions
        for _, prediction in predictions.iterrows():
            self._draw_detection(patch, prediction, is_prediction=True)
        # draw ground truths
        if ground_truths is not None:
            for _, ground_truth in ground_truths.iterrows():
                self._draw_detection(patch, ground_truth, is_prediction=False)
        # postprocess image
        patch = self._final_patch(patch)
        # save files
        video_save_dir = self.save_dir / str(video_id)
        if self.cfg.save_images:
            filepath = video_save_dir / "images" / Path(image_metadata.file_path).name
            filepath.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(filepath), patch)
        if self.cfg.save_videos:
            self._update_video(patch, video_save_dir, video_id)

    def _draw_detection(self, patch, detection, is_prediction):
        # colors
        color_bbox, color_text, color_keypoint, color_skeleton = self._colors(
            detection, is_prediction
        )
        # bbox
        bbox = detection.bbox_ltrb
        bbox = np.round(clip_bbox_ltrb_to_img_dim(bbox, patch.shape[1], patch.shape[0])).astype(
            int
        )
        # bpbreid heatmap (draw before other elements, so that they are not covered by heatmap)
        if is_prediction and self.cfg.prediction.draw_bpbreid_heatmaps:
            img_crop = patch[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            body_masks = detection.body_masks
            img_crop_with_mask = overlay_heatmap(img_crop, body_masks[0], rgb=True)
            patch[bbox[1] : bbox[3], bbox[0] : bbox[2]] = img_crop_with_mask
        if (is_prediction and self.cfg.prediction.draw_bbox) or (
            not is_prediction and self.cfg.ground_truth.draw_bbox
        ):
            cv2.rectangle(
                patch,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=color_bbox,
                thickness=self.cfg.bbox.thickness,
                lineType=cv2.LINE_AA,
            )
        # keypoints
        keypoints = detection.keypoints_xyc
        keypoints_xy = keypoints[:, :2].astype(int)
        keypoints_c = keypoints[:, 2]
        for xy, c in zip(keypoints_xy, keypoints_c):
            if (is_prediction and self.cfg.prediction.draw_keypoints) or (
                not is_prediction and self.cfg.ground_truth.draw_keypoints
            ):
                cv2.circle(
                    patch,
                    (xy[0], xy[1]),
                    color=color_keypoint,
                    radius=self.cfg.keypoint.radius,
                    thickness=self.cfg.keypoint.thickness,
                    lineType=cv2.LINE_AA,
                )
            # confidence
            if (is_prediction and self.cfg.prediction.print_keypoints_confidence) or (
                not is_prediction and self.cfg.ground_truth.print_keypoints_confidence
            ):
                cv2.putText(
                    patch,
                    f"{c:.2}",
                    (xy[0] + 2, xy[1] - 5),
                    fontFace=self.cfg.text.font,
                    fontScale=self.cfg.text.scale,
                    thickness=self.cfg.text.thickness,
                    color=color_text,
                    lineType=cv2.LINE_AA,
                )
        # skeleton
        if (is_prediction and self.cfg.prediction.draw_skeleton) or (
            not is_prediction and self.cfg.ground_truth.draw_skeleton
        ):
            for bone in human_skeleton:
                if keypoints_c[bone[0] - 1] > 0 and keypoints_c[bone[1] - 1] > 0:
                    cv2.line(
                        patch,
                        (keypoints_xy[bone[0] - 1, 0], keypoints_xy[bone[0] - 1, 1],),
                        (keypoints_xy[bone[1] - 1, 0], keypoints_xy[bone[1] - 1, 1],),
                        color=color_skeleton,
                        thickness=self.cfg.skeleton.thickness,
                        lineType=cv2.LINE_AA,
                    )
        # id
        if (is_prediction and self.cfg.prediction.print_id) or (
            not is_prediction and self.cfg.ground_truth.print_id
        ):
            cv2.putText(
                patch,
                f"ID: {detection.track_id}",
                (bbox[0] + 2, bbox[1] - 5),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color=color_text,
                lineType=cv2.LINE_AA,
            )
        # confidence
        if (is_prediction and self.cfg.prediction.print_bbox_confidence) or (
            not is_prediction and self.cfg.ground_truth.print_bbox_confidence
        ):
            cv2.putText(
                patch,
                f"Conf.: {np.mean(keypoints_c):.2}",
                (bbox[0] + 2, bbox[3] - 5),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color=color_text,
                lineType=cv2.LINE_AA,
            )

    def _colors(self, detection, is_prediction):
        if is_prediction:
            if pd.isna(detection.track_id):
                color_bbox = self.cfg.bbox.color_untracked
                color_text = self.cfg.text.color_untracked
                color_keypoint = self.cfg.keypoint.color_untracked
                color_skeleton = self.cfg.skeleton.color_untracked
            else:
                color_id = cmap[int(detection.track_id) % len(cmap)]
                color_bbox = (
                    self.cfg.bbox.color_tracked
                    if self.cfg.bbox.color_tracked
                    else color_id
                )
                color_text = (
                    self.cfg.text.color_tracked
                    if self.cfg.text.color_tracked
                    else color_id
                )
                color_keypoint = (
                    self.cfg.keypoint.color_tracked
                    if self.cfg.keypoint.color_tracked
                    else color_id
                )
                color_skeleton = (
                    self.cfg.skeleton.color_tracked
                    if self.cfg.skeleton.color_tracked
                    else color_id
                )
        else:
            color_bbox = self.cfg.bbox.color_ground_truth
            color_text = self.cfg.text.color_ground_truth
            color_keypoint = self.cfg.keypoint.color_ground_truth
            color_skeleton = self.cfg.skeleton.color_ground_truth
        return color_bbox, color_text, color_keypoint, color_skeleton

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
