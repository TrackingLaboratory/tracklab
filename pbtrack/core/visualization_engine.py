from multiprocessing import Process

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from pbtrack.engine import TrackingEngine
from pbtrack.callbacks import Callback
from pbtrack.utils.cv2_utils import draw_text
from pbtrack.utils.images import cv2_load_image, overlay_heatmap
from pbtrack.utils.coordinates import (
    clip_bbox_ltrb_to_img_dim,
    round_bbox_coordinates,
    bbox_ltwh2ltrb,
)

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

# FIXME can be cleaned and drawing code should be moved to utils folder
class VisualizationEngine(Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = Path("visualization")
        self.processed_video_counter = 0
        self.process = None

    def on_video_loop_end(self, engine: TrackingEngine, video, video_idx, detections):
        # if self.process:
        #     self.process.join()
        # self.process = Process(target=self.run, args=(engine.tracker_state, video_idx))
        # self.process.start()
        self.run(engine.tracker_state, video_idx)

    def run(self, tracker_state, video_id):
        # check for process max video
        if not self.cfg.save_videos and not self.cfg.save_images:
            return
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
        # if self.cfg.resize is not None:
        #     raise NotImplementedError("Resize is not fully yet, bbox/keypoints need to be resized to...")
        #     patch = cv2.resize(patch, self.cfg.resize, interpolation=cv2.INTER_CUBIC)
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
        # draw image metadata
        self._draw_image_metadata(patch, image_metadata)
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
        if (
            "ignore_regions_x" in image_metadata
            and "ignore_regions_y" in image_metadata
        ):
            for x, y in zip(
                image_metadata["ignore_regions_x"], image_metadata["ignore_regions_y"]
            ):
                points = np.array([x, y]).astype(int).T
                points = points.reshape((-1, 1, 2))
                cv2.polylines(
                    patch, [points], True, [255, 0, 0], 2, lineType=cv2.LINE_AA
                )

    def _draw_detection(self, patch, detection, is_prediction):
        is_matched = pd.notna(detection.track_id)
        # colors
        color_bbox, color_text, color_keypoint, color_skeleton = self._colors(
            detection, is_prediction
        )
        bbox_ltrb = clip_bbox_ltrb_to_img_dim(
            round_bbox_coordinates(detection.bbox_ltrb), patch.shape[1], patch.shape[0]
        )
        l, t, r, b = bbox_ltrb
        w, h = r - l, b - t
        # bpbreid heatmap (draw before other elements, so that they are not covered by heatmap)
        if is_prediction and self.cfg.prediction.draw_bpbreid_heatmaps:
            img_crop = patch[bbox_ltrb[1] : bbox_ltrb[3], bbox_ltrb[0] : bbox_ltrb[2]]
            body_masks = detection.body_masks
            img_crop_with_mask = overlay_heatmap(
                img_crop,
                body_masks[0],
                mask_threshold=self.cfg.prediction.heatmaps_display_threshold,
                rgb=True,
            )
            patch[
                bbox_ltrb[1] : bbox_ltrb[3], bbox_ltrb[0] : bbox_ltrb[2]
            ] = img_crop_with_mask
        # bbox
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
        # kf bbox
        if (
            is_prediction
            and self.cfg.prediction.draw_kf_bbox
            and hasattr(detection, "track_bbox_pred_kf_ltwh")
            and not pd.isna(detection.track_bbox_pred_kf_ltwh)
        ):
            # FIXME kf bbox from tracklets that were not matched are not displayed
            bbox_kf_ltrb = clip_bbox_ltrb_to_img_dim(
                round_bbox_coordinates(
                    bbox_ltwh2ltrb(detection.track_bbox_pred_kf_ltwh)
                ),
                patch.shape[1],
                patch.shape[0],
            )
            cv2.rectangle(
                patch,
                (bbox_kf_ltrb[0], bbox_kf_ltrb[1]),
                (bbox_kf_ltrb[2], bbox_kf_ltrb[3]),
                color=self.cfg.bbox.color_kf,
                thickness=self.cfg.bbox.thickness,
                lineType=cv2.LINE_AA,
            )
            draw_text(
                patch,
                f"{int(detection.track_id)}",
                (bbox_kf_ltrb[0] + 3, bbox_kf_ltrb[1] + 3),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color_txt=(50, 50, 50),
                lineType=cv2.LINE_AA,
                color_bg=(255, 255, 255),
                alignV="t",
            )
        # keypoints
        keypoints = detection.keypoints_xyc
        keypoints_xy = keypoints[:, :2].astype(int)
        keypoints_c = keypoints[:, 2]
        for xy, c in zip(keypoints_xy, keypoints_c):
            if c <= 0:
                continue
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
                (int(l + w / 2), t - 5),
                fontFace=self.cfg.text.font + 1,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color=color_text,
                lineType=cv2.LINE_AA,
            )
        # confidence
        if (is_prediction and self.cfg.prediction.print_bbox_confidence) or (
            not is_prediction and self.cfg.ground_truth.print_bbox_confidence
        ):
            draw_text(
                patch,
                f"{detection.keypoints_score:.1f} %",
                (l + 3, t - 5),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color_txt=(0, 0, 255),
                lineType=cv2.LINE_AA,
                color_bg=(255, 255, 255),
            )
        # track state + hits + age
        if (
            is_prediction
            and self.cfg.prediction.print_bbox_confidence
            and is_matched
            and hasattr(detection, "state")
            and hasattr(detection, "hits")
            and hasattr(detection, "age")
        ):
            draw_text(
                patch,
                f"st={detection.state} | #d={detection.hits} | age={detection.age}",
                (r - 3, t + 5),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color_txt=(0, 0, 255),
                lineType=cv2.LINE_AA,
                color_bg=(255, 255, 255),
                alignV="t",
                alignH="r",
            )
        # display_matched_with
        if is_prediction and self.cfg.prediction.display_matched_with:
            if (
                hasattr(detection, "matched_with")
                and detection.matched_with is not None
                and is_matched
            ):
                draw_text(
                    patch,
                    f"{detection.matched_with[0]}|{detection.matched_with[1]:.2f}",
                    (l + 3, t + 5),
                    fontFace=self.cfg.text.font,
                    fontScale=self.cfg.text.scale,
                    thickness=self.cfg.text.thickness,
                    color_txt=(255, 0, 0),
                    color_bg=(255, 255, 255),
                    lineType=cv2.LINE_AA,
                    alignV="t",
                )
        # display_n_closer_tracklets_costs
        if is_prediction and self.cfg.prediction.display_n_closer_tracklets_costs > 0:
            if hasattr(detection, "matched_with") and is_matched:
                nt = self.cfg.prediction.display_n_closer_tracklets_costs
                if "R" in detection.costs:
                    sorted_reid_costs = sorted(
                        list(detection.costs["R"].items()), key=lambda x: x[1]
                    )
                    processed_reid_costs = {
                        t[0]: np.around(t[1], 2) for t in sorted_reid_costs[:nt]
                    }
                    draw_text(
                        patch,
                        f"R({detection.costs['Rt']:.2f}): {processed_reid_costs}",
                        (l + 5, b - 5),
                        fontFace=self.cfg.text.font,
                        fontScale=self.cfg.text.scale,
                        thickness=self.cfg.text.thickness,
                        lineType=cv2.LINE_AA,
                        color_txt=(255, 0, 0),
                        color_bg=(255, 255, 255),
                    )
                if "S" in detection.costs:
                    sorted_st_costs = sorted(
                        list(detection.costs["S"].items()), key=lambda x: x[1]
                    )
                    processed_st_costs = {
                        t[0]: np.around(t[1], 2) for t in sorted_st_costs[:nt]
                    }
                    draw_text(
                        patch,
                        f"S({detection.costs['St']:.2f}): {processed_st_costs}",
                        (l + 5, b - 20),
                        fontFace=self.cfg.text.font,
                        fontScale=self.cfg.text.scale,
                        thickness=self.cfg.text.thickness,
                        lineType=cv2.LINE_AA,
                        color_txt=(0, 255, 0),
                        color_bg=(255, 255, 255),
                    )
                if "K" in detection.costs:
                    sorted_gated_kf_costs = sorted(
                        list(detection.costs["K"].items()), key=lambda x: x[1]
                    )
                    processed_gated_kf_costs = {
                        t[0]: np.around(t[1], 2) for t in sorted_gated_kf_costs[:nt]
                    }
                    draw_text(
                        patch,
                        f"K({detection.costs['Kt']:.2f}): {processed_gated_kf_costs}",
                        (l + 5, b - 35),
                        fontFace=self.cfg.text.font,
                        fontScale=self.cfg.text.scale,
                        thickness=self.cfg.text.thickness,
                        lineType=cv2.LINE_AA,
                        color_txt=(0, 0, 255),
                        color_bg=(255, 255, 255),
                    )
        if (
            is_prediction
            and self.cfg.prediction.display_reid_visibility_scores
            and hasattr(detection, "visibility_scores")
        ):
            draw_text(
                patch,
                f"S: {np.around(detection.visibility_scores.astype(float), 1)}",
                (l + 5, b - 50),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                lineType=cv2.LINE_AA,
                color_txt=(0, 0, 0),
                color_bg=(255, 255, 255),
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

    def _draw_image_metadata(self, patch, image_metadata):
        cv2.putText(
            patch,
            f"{image_metadata.frame}/{image_metadata.nframes}",
            (6, patch.shape[0] - 6),
            fontFace=1,
            fontScale=1.0,
            thickness=1,
            color=(255, 0, 0),
            lineType=cv2.LINE_AA,
        )
