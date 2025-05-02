from .visualizer import ImageVisualizer
from tracklab.utils.cv2 import print_count_frame, draw_ignore_region

class FrameCount(ImageVisualizer):
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        print_count_frame(image, image_gt.frame, nframes=image_gt.nframes)

class IgnoreRegions(ImageVisualizer):
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        draw_ignore_region(image, image_pred)
