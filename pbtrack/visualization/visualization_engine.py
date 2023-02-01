import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from pbtrack.utils.coordinates import clip_bbox_ltrb_to_img_dim, round_bbox_ccordinates
from pbtrack.utils.images import cv2_load_image, overlay_heatmap

ground_truth_cmap = [
    [251, 248, 204],
    [253, 228, 207],
    [255, 207, 210],
    [241, 192, 232],
    [207, 186, 240],
    [163, 196, 243],
    [144, 219, 244],
    [142, 236, 245],
    [152, 245, 225],
    [185, 251, 192],
]

prediction_cmap = [
    [255, 0, 0],
    [255, 135, 0],
    [255, 211, 0],
    [222, 255, 10],
    [161, 255, 10],
    [10, 255, 153],
    [10, 239, 255],
    [20, 125, 245],
    [88, 10, 255],
    [190, 10, 255],
]

posetrack_human_skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 6],
    [2, 7],
    [2, 3],
    [1, 2],
    [1, 3],
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
        # draw ignore regions
        if self.cfg.ground_truth.draw_ignore_region:
            self._draw_ignore_region(patch, image_metadata)
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
        if self.cfg.save_images:
            filepath = (
                self.save_dir
                / "images"
                / str(video_id)
                / Path(image_metadata.file_path).name
            )
            filepath.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(filepath), patch)
        if self.cfg.save_videos:
            self._update_video(patch, video_id)

    def _draw_ignore_region(self, patch, image_metadata):
        if 'ignore_regions_x' in image_metadata and 'ignore_regions_y' in image_metadata:
            for (x, y) in zip(
                image_metadata["ignore_regions_x"], image_metadata["ignore_regions_y"]
            ):
                points = np.array([x, y]).astype(int).T
                points = points.reshape((-1, 1, 2))
                cv2.polylines(patch, [points], True, [255, 0, 0], 2, lineType=cv2.LINE_AA)

    def _draw_detection(self, patch, detection, is_prediction):
        # colors
        color_bbox, color_text, color_keypoint, color_skeleton = self._colors(
            detection, is_prediction
        )
        # bbox
        bbox_ltrb = clip_bbox_ltrb_to_img_dim(
            round_bbox_ccordinates(detection.bbox_ltrb), patch.shape[1], patch.shape[0]
        )
        # bpbreid heatmap (draw before other elements, so that they are not covered by heatmap)
        if is_prediction and self.cfg.prediction.draw_bpbreid_heatmaps:
            img_crop = patch[bbox_ltrb[1] : bbox_ltrb[3], bbox_ltrb[0] : bbox_ltrb[2]]
            body_masks = detection.body_masks
            img_crop_with_mask = overlay_heatmap(img_crop, body_masks[0], rgb=True)
            patch[
                bbox_ltrb[1] : bbox_ltrb[3], bbox_ltrb[0] : bbox_ltrb[2]
            ] = img_crop_with_mask
        if (is_prediction and self.cfg.prediction.draw_bbox) or (
            not is_prediction and self.cfg.ground_truth.draw_bbox
        ):
            cv2.rectangle(
                patch,
                (bbox_ltrb[0], bbox_ltrb[1]),
                (bbox_ltrb[2], bbox_ltrb[3]),
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
                    f"{100*c:.1f} %",
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
            for link in posetrack_human_skeleton:
                if keypoints_c[link[0] - 1] > 0 and keypoints_c[link[1] - 1] > 0:
                    cv2.line(
                        patch,
                        (
                            keypoints_xy[link[0] - 1, 0],
                            keypoints_xy[link[0] - 1, 1],
                        ),
                        (
                            keypoints_xy[link[1] - 1, 0],
                            keypoints_xy[link[1] - 1, 1],
                        ),
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
                "nan"
                if pd.isna(detection.track_id)
                else f"{int(detection.track_id)}",  # FIXME why detection.track_id is float ?
                (bbox_ltrb[0] + 2, bbox_ltrb[1] - 5),
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
                f"{100*np.mean(keypoints_c):.1f} %",
                (bbox_ltrb[0] + 2, bbox_ltrb[3] - 5),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color=color_text,
                lineType=cv2.LINE_AA,
            )

    def _colors(self, detection, is_prediction):
        cmap = prediction_cmap if is_prediction else ground_truth_cmap
        if pd.isna(detection.track_id):
            color_bbox = self.cfg.bbox.color_no_id
            color_text = self.cfg.text.color_no_id
            color_keypoint = self.cfg.keypoint.color_no_id
            color_skeleton = self.cfg.skeleton.color_no_id
        else:
            color_key = "color_prediction" if is_prediction else "color_ground_truth"
            color_id = cmap[int(detection.track_id) % len(cmap)]
            color_bbox = (
                self.cfg.bbox[color_key] if self.cfg.bbox[color_key] else color_id
            )
            color_text = (
                self.cfg.text[color_key] if self.cfg.text[color_key] else color_id
            )
            color_keypoint = (
                self.cfg.keypoint[color_key]
                if self.cfg.keypoint[color_key]
                else color_id
            )
            color_skeleton = (
                self.cfg.skeleton[color_key]
                if self.cfg.skeleton[color_key]
                else color_id
            )
        return color_bbox, color_text, color_keypoint, color_skeleton

    def _update_video(self, patch, video_id):
        if not hasattr(self, "video_writer"):
            filepath = self.save_dir / "videos" / f"{video_id}.mp4"
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
