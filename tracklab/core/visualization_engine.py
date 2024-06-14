import platform

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm

from tracklab.datastruct import TrackerState
from tracklab.callbacks import Callback
from tracklab.utils.cv2 import (
    draw_text,
    draw_bbox,
    draw_bpbreid_heatmaps,
    draw_keypoints,
    draw_ignore_region,
    final_patch,
    print_count_frame,
    cv2_load_image,
)
from matplotlib import colormaps

# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.coordinates import (
    clip_bbox_ltrb_to_img_dim,
    round_bbox_coordinates,
    bbox_ltwh2ltrb,
)

import logging

from tracklab.utils.progress import progress

log = logging.getLogger(__name__)

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

left_cmap = colormaps["Blues"].reversed().resampled(100)
right_cmap = colormaps["Reds"].reversed().resampled(100)


class VisualizationEngine(Callback):
    after_saved_state = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = Path("visualization")
        self.processed_video_counter = 0
        self.process = None
        self.video_name = None
        self.windows = []

    def on_image_loop_end(
        self,
        engine: "TrackingEngine",
        image_metadata: pd.Series,
        image,
        image_idx: int,
        detections: pd.DataFrame,
    ):
        if self.cfg.show_online:
            tracker_state = engine.tracker_state
            if tracker_state.detections_gt is not None:
                ground_truths = tracker_state.detections_gt[
                    tracker_state.detections_gt.image_id == image_metadata.name
                ]
            else:
                ground_truths = None
            if len(detections) == 0:
                image = image
            else:
                detections = detections[detections.image_id == image_metadata.name]
                image = self.draw_frame(image_metadata,
                                        detections, ground_truths, "inf", image=image)
            if platform.system() == "Linux" and self.video_name not in self.windows:
                self.windows.append(self.video_name)
                cv2.namedWindow(str(self.video_name),
                                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(self.video_name), image.shape[1], image.shape[0])
            cv2.imshow(str(self.video_name), image)
            cv2.waitKey(1)

    def on_video_loop_start(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,  # FIXME keep ?
        # image_metadatas: pd.DataFrame,  # FIXME add ?
        video_idx: int,
        index: int,  # FIXME change name ?
    ):
        self.video_name = video_metadata.name

    def on_video_loop_end(self, engine, video_metadata, video_idx, detections, image_pred):
        if self.cfg.save_videos or self.cfg.save_images:
            if (
                self.processed_video_counter < self.cfg.process_n_videos
                or self.cfg.process_n_videos == -1
            ):
                if "progress" in engine.callbacks:
                    progress = engine.callbacks["progress"]
                else:
                    progress = None
                self.run(engine.tracker_state, video_idx, detections, image_pred, progress=progress)

    def run(self, tracker_state: TrackerState, video_id, detections, image_preds, progress=None):
        image_metadatas = tracker_state.image_metadatas[
            tracker_state.image_metadatas.video_id == video_id
        ]
        image_gts = tracker_state.image_gt[tracker_state.image_gt.video_id == video_id]
        nframes = len(image_metadatas)
        video_name = tracker_state.video_metadatas.loc[video_id].name
        if progress:
            if self.cfg.process_n_frames_by_video == -1:
                total = len(image_metadatas.index)
            else:
                total = self.cfg.process_n_frames_by_video
            progress.init_progress_bar("vis", "Visualization", total)
        for i, image_id in enumerate(image_metadatas.index):
            # check for process max frame per video
            if i >= self.cfg.process_n_frames_by_video != -1:
                break
            # retrieve results
            image_metadata = image_metadatas.loc[image_id]
            image_gt = image_gts.get(image_id, None)
            image_pred = image_preds.loc[image_id]
            detections_pred = detections[
                detections.image_id == image_metadata.name
            ]
            if tracker_state.detections_gt is not None:
                ground_truths = tracker_state.detections_gt[
                    tracker_state.detections_gt.image_id == image_metadata.name
                ]
            else:
                ground_truths = None
            # process the detections
            self._process_frame(
                image_metadata, detections_pred, ground_truths, video_name, nframes,
                image_pred, image_gt
            )
            if progress:
                progress.on_module_step_end(None, "vis", None, None)
        # save the final video
        if self.cfg.save_videos:
            self.video_writer.release()
            delattr(self, "video_writer")
        self.processed_video_counter += 1
        if progress:
            progress.on_module_end(None, "vis", None)

    def _process_frame(
        self, image_metadata, detections_pred, ground_truths, video_name, nframes,
            image_pred, image_gt
    ):
        patch = self.draw_frame(image_metadata, detections_pred, ground_truths, image_pred, image_gt, nframes)
        # save files
        if self.cfg.save_images:
            filepath = (
                self.save_dir
                / "images"
                / str(video_name)
                / Path(image_metadata.file_path).name
            )
            filepath.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(filepath), patch)
        if self.cfg.save_videos:
            self._update_video(patch, video_name)

    def draw_frame(self, image_metadata, detections_pred, ground_truths, image_pred, image_gt, nframes, image=None):
        if image is not None:
            patch = image
        else:
            patch = cv2_load_image(image_metadata.file_path)

        # print count of frame
        print_count_frame(patch, image_metadata.frame, nframes)

        # draw ignore regions
        if self.cfg.ground_truth.draw_ignore_region:
            draw_ignore_region(patch, image_metadata)

        # draw detections_pred
        for _, detection_pred in detections_pred.iterrows():
            self._draw_detection(patch, detection_pred, is_prediction=True)

        # draw ground truths
        if ground_truths is not None:
            for _, ground_truth in ground_truths.iterrows():
                self._draw_detection(patch, ground_truth, is_prediction=False)

        # postprocess image
        patch = final_patch(patch)
        return patch

    def _draw_detection(self, patch, detection, is_prediction):
        is_matched = pd.notna(detection.track_id)

        if not is_matched and not self.cfg.prediction.draw_unmatched:
            return

        # colors
        color_bbox, color_text, color_keypoint, color_skeleton = self._colors(
            detection, is_prediction
        )

        # bpbreid heatmap (draw before other elements, so that they are not covered by heatmap)
        if is_prediction and self.cfg.prediction.draw_bpbreid_heatmaps:
            draw_bpbreid_heatmaps(
                detection, patch, self.cfg.prediction.heatmaps_display_threshold
            )

        # bbox, confidence, id
        if (is_prediction and self.cfg.prediction.draw_bbox) or (
            not is_prediction and self.cfg.ground_truth.draw_bbox
        ):
            print_confidence = (
                is_prediction and self.cfg.prediction.print_bbox_confidence
            ) or (not is_prediction and self.cfg.ground_truth.print_bbox_confidence)
            print_id = (is_prediction and self.cfg.prediction.print_id) or (
                not is_prediction and self.cfg.ground_truth.print_id
            )
            draw_bbox(
                detection,
                patch,
                color_bbox,
                self.cfg.bbox.thickness,
                self.cfg.text.font,
                self.cfg.text.scale,
                self.cfg.text.thickness,
                color_text,
                print_confidence,
                print_id,
            )

        # keypoints, confidences, skeleton
        if "keypoints_xyc" in detection and (is_prediction and self.cfg.prediction.draw_keypoints) or (
            not is_prediction and self.cfg.ground_truth.draw_keypoints
        ):
            print_confidence = (
                is_prediction and self.cfg.prediction.print_keypoints_confidence
            ) or (
                not is_prediction and self.cfg.ground_truth.print_keypoints_confidence
            )
            draw_skeleton = (is_prediction and self.cfg.prediction.draw_skeleton) or (
                not is_prediction and self.cfg.ground_truth.draw_skeleton
            )
            detection.keypoints_xyc[detection.keypoints_xyc[:, 2] < self.cfg.vis_kp_threshold] = 0.

            draw_keypoints(
                detection,
                patch,
                color_keypoint,
                self.cfg.keypoint.radius,
                self.cfg.keypoint.thickness,
                self.cfg.text.font,
                self.cfg.text.scale,
                self.cfg.text.thickness,
                color_text,
                color_skeleton,
                self.cfg.skeleton.thickness,
                print_confidence,
                draw_skeleton,
            )

        # FIXME clean, put try catch, move to utils/cv2.py
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
                color_bg=(255, 255, 255),
                alignV="t",
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
            l, t, r, b = detection.bbox.ltrb(
                image_shape=(patch.shape[1], patch.shape[0]), rounded=True
            )
            draw_text(
                patch,
                f"st={detection.state} | #d={detection.hits} | age={detection.age}",
                (r - 3, t + 5),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
                color_txt=(0, 0, 255),
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
                l, t, r, b = detection.bbox.ltrb(
                    image_shape=(patch.shape[1], patch.shape[0]), rounded=True
                )
                draw_text(
                    patch,
                    f"{detection.matched_with[0]}|{detection.matched_with[1]:.2f}",
                    (r-3, t + 20),
                    fontFace=self.cfg.text.font,
                    fontScale=self.cfg.text.scale,
                    thickness=self.cfg.text.thickness,
                    color_txt=(255, 0, 0),
                    color_bg=(255, 255, 255),
                    alignV="t",
                    alignH="r",
                )
        # display_n_closer_tracklets_costs
        if is_prediction and self.cfg.prediction.display_n_closer_tracklets_costs > 0:
            l, t, r, b = detection.bbox.ltrb(
                image_shape=(patch.shape[1], patch.shape[0]), rounded=True
            )
            if hasattr(detection, "matched_with") and is_matched:
                nt = self.cfg.prediction.display_n_closer_tracklets_costs
                if "R" in detection.costs:
                    sorted_reid_costs = sorted(
                        list(detection.costs["R"].items()), key=lambda x: x[1], reverse=True
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
                        color_txt=(150, 0, 0),
                        color_bg=(255, 255, 255),
                    )
                if "S" in detection.costs:
                    sorted_st_costs = sorted(
                        list(detection.costs["S"].items()), key=lambda x: x[1], reverse=True
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
                        color_txt=(0, 150, 0),
                        color_bg=(255, 255, 255),
                    )
                if "K" in detection.costs:
                    sorted_gated_kf_costs = sorted(
                        list(detection.costs["K"].items()), key=lambda x: x[1], reverse=True
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
                        color_txt=(0, 0, 150),
                        color_bg=(255, 255, 255),
                    )

        # display visibility_scores
        if (
            is_prediction
            and self.cfg.prediction.display_reid_visibility_scores
            and hasattr(detection, "visibility_scores")
        ):
            l, t, r, b = detection.bbox.ltrb(
                image_shape=(patch.shape[1], patch.shape[0]), rounded=True
            )
            draw_text(
                patch,
                f"S: {np.around(detection.visibility_scores.astype(float), 1)}",
                (l + 5, b - 50),
                fontFace=self.cfg.text.font,
                fontScale=self.cfg.text.scale,
                thickness=self.cfg.text.thickness,
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
                self.cfg.bbox[color_key] if self.cfg.bbox[color_key] is not None else color_id
            )
            color_text = (
                self.cfg.text[color_key] if self.cfg.text[color_key] is not None else color_id
            )
            color_keypoint = (
                self.cfg.keypoint[color_key]
                if self.cfg.keypoint[color_key] is not None
                else color_id
            )
            color_skeleton = (
                self.cfg.skeleton[color_key]
                if self.cfg.skeleton[color_key] is not None
                else color_id
            )
        return color_bbox, color_text, color_keypoint, color_skeleton

    def _update_video(self, patch, video_name):
        if not hasattr(self, "video_writer"):
            filepath = self.save_dir / "videos" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.cfg.video_fps),
                (patch.shape[1], patch.shape[0]),
            )
        self.video_writer.write(patch)
