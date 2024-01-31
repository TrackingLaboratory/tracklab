import cv2
import numpy as np

from sn_calibration_baseline.soccerpitch import SoccerPitch
from tracklab.utils.cv2 import draw_text


def draw_pitch(patch, detections_pred, detections_gt,
               image_pred, image_gt,
               line_thickness=3,
               radar_view_scale=3,
               pitch_image=None,
               ):

    # Draw the lines on the image pitch
    image_height, image_width, _ = patch.shape
    for name, line in image_pred["lines"].items():
        for j in np.arange(len(line)-1):
            cv2.line(
                patch,
                (int(line[j]["x"] * image_width), int(line[j]["y"] * image_height)),
                (int(line[j+1]["x"] * image_width), int(line[j+1]["y"] * image_height)),
                color=SoccerPitch.palette[name],
                thickness=line_thickness,  # TODO : make this a parameter
            )

    # Draw the Top-view pitch
    draw_radar_view(patch, detections_gt, scale=radar_view_scale, group="gt", pitch_image=pitch_image)
    draw_radar_view(patch, detections_pred, scale=radar_view_scale, group="pred", pitch_image=pitch_image)


def draw_radar_view(patch, detections, scale, delta=32, group="gt", pitch_image=None):
    pitch_width = 105 + 2 * 10  # pitch size + 2 * margin
    pitch_height = 68 + 2 * 5  # pitch size + 2 * margin
    sign = -1 if group == "gt" else +1
    radar_center_x = int(1920/2 - pitch_width * scale / 2 * sign - delta * sign)
    radar_center_y = int(1080 - pitch_height * scale / 2)
    radar_top_x = int(radar_center_x - pitch_width * scale / 2)
    radar_top_y = int(1080 - pitch_height * scale)
    radar_width = int(pitch_width * scale)
    radar_height = int(pitch_height * scale)
    if pitch_image is not None:
        radar_img = cv2.resize(cv2.imread(pitch_image), (pitch_width * scale, pitch_height * scale))
        radar_img = cv2.bitwise_not(radar_img)
    else:
        radar_img = np.ones((pitch_height * scale, pitch_width * scale, 3)) * 255
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :] = radar_img
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :] = radar_img
    draw_text(
        patch,
        group,
        (radar_center_x, radar_top_y),
        1, 1, 1,
        color_txt=(255, 255, 255),
        alignH="c",
        alignV="b",
    )
    for name, detection in detections.iterrows():
        if "role" in detection and detection.role == "ball":
            continue
        if "role" in detection and "team" in detection:
            color = (0, 0, 255) if detection.team == "left" else (255, 0, 0)
        else:
            color = (0, 0, 0)
        bbox_name = "bbox_pitch"
        if not isinstance(detection[bbox_name], dict):
            continue
        x_middle = np.clip(detection[bbox_name]["x_bottom_middle"], -10000, 10000)
        y_middle = np.clip(detection[bbox_name]["y_bottom_middle"], -10000, 10000)
        cat = None
        if "jersey_number" in detection and detection.jersey_number is not None:
            cat = f"{detection.jersey_number:02}"
        elif "role" in detection:
            if detection.role == "goalkeeper":
                cat = "GK"
            elif detection.role == "referee":
                cat = "RE"
                color = (238, 210, 2)
        if cat is not None:
            draw_text(
                patch,
                cat,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                1,
                0.6,
                1,
                color_txt=color,
                alignH="c",
                alignV="c",
            )
        else:
            cv2.circle(
                patch,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                scale,
                color=color,
                thickness=-1
            )