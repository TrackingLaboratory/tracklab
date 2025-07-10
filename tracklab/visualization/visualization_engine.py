from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional
import logging

import cv2
import pandas as pd
import numpy as np
import platform

from tracklab.callbacks import Progressbar, Callback
from tracklab.visualization import Visualizer
from tracklab.datastruct import TrackerState
from tracklab.utils.cv2 import final_patch, cv2_load_image

log = logging.getLogger(__name__)

class VisualizationEngine(Callback):
    """ Visualization engine from list of visualizers.

    Args:
        visualizers: a list of visualizer instances, which must implement `draw_frame`,
                     or subclass :class:`DetectionVisualizer` and implement
                     `draw_detection`.
        save_images: whether to save the visualization as image files (.jpeg)
        save_videos: whether to save the visualization as video files (.mp4)
        show_online: whether to show online tracking in realtime (only work if the pipeline doesn't involve VideoLevelModule)
        process_n_videos: number of videos to visualize. Will visualize the first N videos.
        process_n_frames_by_video: number of frames per video to visualize. Will visualize
                                   frames every N/n frames (not first n frames)
    """

    def __init__(self,
                 visualizers: Dict[str, Visualizer],
                 save_images: bool = False,
                 save_videos: bool = False,
                 show_online: bool = False,
                 video_fps: int = 25,
                 process_n_videos: Optional[int] = None,
                 process_n_frames_by_video: Optional[int] = None,
                 **kwargs
                 ):
        self.visualizers = visualizers
        self.save_dir = Path("visualization")
        self.save_images = save_images
        self.save_videos = save_videos
        self.show_online = show_online
        self.video_fps = video_fps
        self.max_videos = process_n_videos
        self.max_frames = process_n_frames_by_video
        self.windows = []
        for visualizer in visualizers.values():
            visualizer.post_init(**kwargs)

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        if self.show_online:
            cv2.destroyAllWindows()
        if self.save_videos or self.save_images:
            log.info(f"Visualization output at : {self.save_dir.absolute()}")

    def on_video_loop_end(self, engine, video_metadata, video_idx, detections,
                          image_pred):
        if self.save_videos or self.save_images:
            progress = engine.callbacks.get("progress", Progressbar(dummy=True))
            self.visualize(engine.tracker_state, video_idx, detections, image_pred, progress)
            progress.on_module_end(None, "vis", None)

    def on_image_loop_end(self, engine, image_metadata, image, image_idx, detections):
        """
        Handle real-time display during online video tracking.
        """
        if not self.show_online:
            return

        try:
            # Filter detections for current frame
            frame_detections = (
                detections[detections.image_id == image_metadata.name]
                if len(detections) > 0
                else pd.DataFrame()
            )

            # Get ground truth (usually None for online tracking)
            ground_truths = pd.DataFrame()

            # Create dummy image metadata for compatibility
            image_pred = pd.Series(
                {
                    "lines": getattr(image_metadata, "lines", {}),
                    "keypoints": getattr(image_metadata, "keypoints", {}),
                    "file_path": f"frame_{image_idx:06d}.jpg",  # Dummy path
                },
                name=image_metadata.name,
            )

            image_gt = pd.Series(
                {
                    "frame": image_idx,
                    "nframes": -1,  # Unknown total frames in online mode
                },
                name=image_metadata.name,
            )

            # Draw frame with all visualizers
            display_image = self.draw_online_frame(
                image_metadata,
                image,
                frame_detections,
                ground_truths,
                image_pred,
                image_gt,
                nframes=-1,
            )

            # Display the image
            video_name = str(engine.video_filename)
            if platform.system() == "Linux" and video_name not in self.windows:
                self.windows.append(video_name)
                cv2.namedWindow(video_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(
                    video_name, display_image.shape[1], display_image.shape[0]
                )

            # Convert RGB to BGR for OpenCV display
            cv2.imshow(video_name, display_image)
            cv2.waitKey(1)  # Non-blocking wait

        except Exception as e:
            log.warning(f"Error in online visualization: {e}")

    def draw_online_frame(
        self,
        image_metadata,
        image,
        detections_pred,
        detections_gt,
        image_pred,
        image_gt,
        nframes,
    ):
        """Draw frame using all configured visualizers."""
        # Create a copy of the image to avoid modifying the original
        image = np.copy(image)

        for visualizer in self.visualizers.values():
            try:
                visualizer.draw_frame(
                    image, detections_pred, detections_gt, image_pred, image_gt
                )
            except Exception as e:
                log.warning(f"Visualizer {type(visualizer).__name__} raised error: {e}")

        return final_patch(image)

    def visualize(self, tracker_state: TrackerState, video_id, detections, image_preds, progress=None):
        image_metadatas = tracker_state.image_metadatas[tracker_state.image_metadatas.video_id == video_id]
        image_gts = tracker_state.image_gt[tracker_state.image_gt.video_id == video_id]
        nframes = len(image_metadatas)
        video_name = tracker_state.video_metadatas.loc[video_id]["name"]
        for visualizer in self.visualizers.values():
            try:
                visualizer.preproces(detections, tracker_state.detections_gt, image_preds, tracker_state.image_gt)
            except Exception as e:
                log.warning(f"Visualizer {Visualizer} raised error : {e} during preprocess.")
        total = self.max_frames or len(image_metadatas.index)
        progress.init_progress_bar("vis", "Visualization", total)
        detection_preds_by_image = detections.groupby("image_id")
        detection_gts_by_image = tracker_state.detections_gt.groupby("image_id")
        args = [create_draw_args(
            image_id,
            self,
            image_metadatas,
            get_group(detection_preds_by_image, image_id),
            get_group(detection_gts_by_image, image_id),
            image_gts,
            image_preds,
            nframes,
        ) for image_id in islice(image_metadatas.index, 0, None, nframes//total)]
        if self.save_videos:
            image = cv2_load_image(image_metadatas.iloc[0].file_path)
            filepath = self.save_dir / "videos" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.video_fps),
                (image.shape[1], image.shape[0]),
            )
        with Pool() as p:
            for output_image, file_name in p.imap(process_frame, args):
                if self.save_images:
                    filepath = self.save_dir / "images" / str(video_name) / file_name
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(filepath), output_image)
                if self.save_videos:
                    video_writer.write(output_image)
                progress.on_module_step_end(None, "vis", None, None)

    def draw_frame(self, image_metadata, detections_pred, detections_gt,
                   image_pred, image_gt, nframes):
        image = cv2_load_image(image_metadata.file_path)
        for visualizer in self.visualizers.values():
            try:
                visualizer.draw_frame(image, detections_pred, detections_gt, image_pred, image_gt)
            except Exception as e:
                log.warning(f"Visualizer {Visualizer} raised error : {e} during drawing.")
        return final_patch(image)


def create_draw_args(image_id, instance, image_metadatas, detections_pred, detections_gt,
                     image_gts, image_preds, nframes):
    image_metadata = image_metadatas.loc[image_id]
    image_gt = image_gts.loc[image_id]
    image_pred = image_preds.loc[image_id]
    return (instance, image_metadata, detections_pred, detections_gt, image_pred,
            image_gt, nframes)


def process_frame(args):
    (instance, image_metadata, detections_pred, detections_gt,
     image_pred, image_gt, nframes) = args
    frame = instance.draw_frame(image_metadata, detections_pred, detections_gt,
                                image_pred, image_gt, nframes)

    return frame, Path(image_metadata.file_path).name


def get_group(g, key):
    if key in g.groups: return g.get_group(key)
    return pd.DataFrame(columns=["bbox_ltwh"])
