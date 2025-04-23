import cv2
import pandas as pd
import colorsys
import matplotlib.cm as cm
import distinctipy

from .coordinates import *
from functools import lru_cache

import logging

log = logging.getLogger(__name__)

# FIXME load from categories in video_metadata
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


class VideoReader:
    def __init__(self):
        self.filename = None
        self.cap = None  # cv2.VideoCapture(filename)

    def set_filename(self, filename):
        self.filename = filename
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.filename)
        assert self.cap.isOpened(), "Error opening video stream or file"

    def __getitem__(self, idx):
        assert self.filename is not None, "You should first set the filename"
        cap = cv2.VideoCapture(self.filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx-1)
        ret, image = cap.read()
        cap.release()
        assert ret, "Read past the end of the video file"
        return image


video_reader = VideoReader()

@lru_cache(maxsize=32)
def cv2_load_image(file_path):
    file_path = str(file_path)
    if file_path.startswith("vid://"):
        file_path = file_path.removeprefix("vid://")
        video_file, frame_id = file_path.split(":")
        if video_reader.filename != video_file:
            video_reader.set_filename(video_file)
        image = video_reader[int(frame_id)]
    else:
        image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def crop_bbox_ltwh(img, bbox):
    bbox = np.array(bbox).astype(int)
    img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    return img
    

def draw_keypoints(
    detection,
    patch,
    kp_color,
    threshold=0.,
    print_confidence=False,
    draw_skeleton=True,
    skeleton_thickness=2,
    kp_radius=4,
    kp_thickness=-1,
    text_font=1,
    text_scale=1,
    text_thickness=1,
):
    if hasattr(detection, "keypoints_xyc"):
        keypoints_xy = detection.keypoints.xy(rounded=True).astype(int)
        keypoints_c = detection.keypoints.c()
        for xy, c in zip(keypoints_xy, keypoints_c):
            if c >= threshold and (xy != [0, 0]).all():
                cv2.circle(
                    patch,
                    (xy[0], xy[1]),
                    color=kp_color,
                    radius=kp_radius,
                    thickness=kp_thickness,
                    lineType=cv2.LINE_AA,
                )
                if print_confidence:
                    draw_text(
                        patch,
                        f"{100 * c:.1f} %",
                        (xy[0], xy[1]),
                        fontFace=text_font,
                        fontScale=text_scale,
                        thickness=text_thickness,
                        color_txt=kp_color,
                        color_bg=(255, 255, 255),
                        alignH="r",
                        alignV="t",
                        alpha_bg=0.5,
                    )
            if draw_skeleton:
                for link in posetrack_human_skeleton:
                    if (keypoints_c[link[0] - 1] >= threshold and keypoints_c[link[1] - 1] >= threshold) \
                        and (keypoints_xy[link[0] - 1] != [0, 0]).all() \
                        and (keypoints_xy[link[1] - 1] != [0, 0]).all():
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
                            color=kp_color,
                            thickness=skeleton_thickness,
                            lineType=cv2.LINE_AA,
                        )
    else:
        log.warning(
            "You tried to draw the keypoints but no 'keypoints_xyc' were found in the "
            "detection."
        )

def draw_bbox(
    detection,
    patch,
    bbox_color,
    print_id=False,
    print_confidence=False,
    bbox_thickness=2,
    text_font=1,
    text_scale=1,
    text_thickness=1,
):
    if hasattr(detection, "bbox_ltwh"):
        l, t, r, b = detection.bbox.ltrb(image_shape=(patch.shape[1], patch.shape[0]), rounded=True)
        cv2.rectangle(
            patch,
            (l, t),
            (r, b),
            color=bbox_color,
            thickness=bbox_thickness,
            lineType=cv2.LINE_AA,
        )
        if print_confidence:
            if hasattr(detection, "bbox_conf"):

                draw_text(
                    patch,
                    f"{detection.bbox.conf() * 100:.1f}%",
                    (l+5, t+5),
                    fontFace=text_font,
                    fontScale=text_scale,
                    thickness=text_thickness,
                    color_txt=bbox_color,
                    alignH="l",
                    alignV="t",
                    color_bg=(255, 255, 255),
                    alpha_bg=0.5,
                )
            else:
                log.warning(
                    "You tried to draw the confidence but no 'bbox_conf' was found in the "
                    "detection."
                )
        if print_id:
            if hasattr(detection, "track_id"):
                if not np.isnan(detection.track_id):
                    text_color = np.array(distinctipy.get_text_color(np.array(bbox_color) / 255, 0.6)) * 255
                    draw_text(
                        patch,
                        f"ID: {int(detection.track_id)}",
                        (r-5, t-15),
                        fontFace=text_font,
                        fontScale=text_scale,
                        thickness=text_thickness,
                        alignH="r",
                        alignV="t",
                        color_txt=text_color,
                        color_bg=bbox_color, # (255, 255, 255),
                        alpha_bg=0.5,
                    )
            else:
                log.warning(
                    "You tried to draw the track id but no 'track_id' was found in the "
                    "detection."
            )

def draw_bbox_stats(
    detection,
    patch,
    stats,
    text_font=1,
    text_scale=0.7,
    text_thickness=1,
):
    if hasattr(detection, "bbox_ltwh"):
        l, t, r, b = detection.bbox.ltrb(image_shape=(patch.shape[1], patch.shape[0]), rounded=True)
        for i, stat in enumerate(stats):
            if hasattr(detection, stat):
                draw_text(
                    patch,
                    f"{stat}: {pretty_print(stat, detection[stat])}",
                    (r - 5, b - 15*(i+1)),
                    fontFace=text_font,
                    fontScale=text_scale,
                    thickness=text_thickness,
                    alignH="r",
                    alignV="t",
                    color_txt=(0,0,0),
                    color_bg=(255, 255, 255),
                    alpha_bg=0.8,
                )
            else:
                log.warning(f"No '{stat}' found in the detection during visualization.")
    else:
        log.warning(f"No 'bbox_ltwh' found in the detection during visualization.")

def pretty_print(stat, value):
    if stat in ["hits", "age", "time_since_update"]:
        return int(value)
    elif stat == "matched_with":
        if pd.isna(value):
            return "N/A"
        else:
            return f"S: {value[1]:.3f}%"
    elif stat == "costs":
        return {k: {ik: round(iv, 3) for ik, iv in v.items()} if isinstance(v, dict) else round(v, 3)
        if isinstance(v, float) else v for k, v in value.items()}
    else:
        return value

def draw_bpbreid_heatmaps(detection, patch, heatmaps_display_threshold):
    try:
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(patch.shape[1], patch.shape[0]), rounded=True
        )
        img_crop = patch[t:b, l:r]
        body_masks = detection.body_masks
        img_crop_with_mask = overlay_heatmap(
            img_crop,
            body_masks[0],
            mask_threshold=heatmaps_display_threshold,
            rgb=True,
        )
        patch[t:b, l:r] = img_crop_with_mask
    except KeyError:
        log.warning(
            "You tried to draw the bpbreid heatmaps but no 'body_masks' were found in "
            "the detection."
        )


def overlay_heatmap(
    img, heatmap, weight=0.5, mask_threshold=0.0, color_map=cv2.COLORMAP_JET, rgb=False
):
    """
    Overlay a heatmap on an image with given color map.
    Args:
        img: input image in OpenCV BGR format
        heatmap: heatmap to overlay in float from range 0->1. Will be resized to image size and values clipped to 0->1.
        weight: alpha blending weight between image and heatmap, 1-weight for image and weight for heatmap. Set to -1 to
        use the heatmap as alpha channel.
        color_map: OpenCV type colormap to use for heatmap.
        mask_threshold: heatmap values below this threshold will not be displayed.
        rgb: if True, the heatmap will be converted to RGB before overlaying.

    Returns:
        Image with heatmap overlayed.
    """
    width, height = img.shape[1], img.shape[0]
    heatmap = cv2.resize(heatmap, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_gray = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, color_map).astype(img.dtype)
    if rgb:
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    mask_alpha = np.ones_like(heatmap)
    mask_alpha[heatmap < mask_threshold] = 0
    mask_alpha = np.repeat(np.expand_dims(mask_alpha, 2), 3, 2)
    if weight == -1:
        weight = np.repeat(np.expand_dims(heatmap / 2, 2), 3, 2)
    img_with_heatmap = (
        img * (1 - mask_alpha * weight) + heatmap_color * mask_alpha * weight
    )
    # img_with_heatmap = cv2.addWeighted(img, 1-weight, heatmap_color.astype(img.dtype), weight, 0)
    return img_with_heatmap


def draw_ignore_region(patch, image_metadata):
    try:
        for x, y in zip(
            image_metadata["ignore_regions_x"], image_metadata["ignore_regions_y"]
        ):
            points = np.array([x, y]).astype(int).T
            points = points.reshape((-1, 1, 2))
            cv2.polylines(patch, [points], True, [255, 0, 0], 2, lineType=cv2.LINE_AA)
    except KeyError:
        log.warning(
            "You tried to draw the ignored regions but no 'ignore_regions_x' or 'ignore_regions_y' were found in "
            "the image metadata."
        )


def print_count_frame(patch, frame, nframes):
    draw_text(
        patch,
        f"{frame+1}/{nframes}",
        (6, 15),
        fontFace=1,
        fontScale=2.0,
        thickness=1,
        color_txt=(255, 0, 0),
        color_bg=(255, 255, 255),
        alignH="l",
        alignV="t",
    )


def final_patch(patch):
    return cv2.cvtColor(patch, cv2.COLOR_RGB2BGR).astype(np.uint8)


def draw_text(
    img,
    text,
    pos,
    fontFace,
    fontScale,
    thickness,
    lineType=cv2.LINE_AA,
    color_txt=(0, 0, 0),
    color_bg=None,
    alpha_bg=1.0,
    alignH="l",  # l: left, c: center, r: right
    alignV="b",  # t: top, c: center, b: bottom
    darken=1.0,
):
    # TODO: add multiline support
    # TODO: add scale: txt size depend on scale of bbox?
    x, y = pos
    text_size, _ = cv2.getTextSize(
        text, fontFace=fontFace, fontScale=fontScale, thickness=thickness
    )
    text_w, text_h = text_size
    if alignV == "b":
        # txt_pos_y = round((y + fontScale - 1))
        txt_pos_y = y
    elif alignV == "t":
        # txt_pos_y = round((y + fontScale - 1)) + text_h
        txt_pos_y = y + text_h
    elif alignV == "c":
        txt_pos_y = y + text_h // 2
    else:
        raise ValueError("alignV must be one of 't', 'b', 'c'")

    if alignH == "l":
        txt_pos_x = x
    elif alignH == "r":
        txt_pos_x = x - text_w
    elif alignH == "c":
        txt_pos_x = x - text_w // 2
    else:
        raise ValueError("alignH must be one of 'l', 'r', 'c'")

    text_position = (txt_pos_x, txt_pos_y)
    padding = 3
    rect_pos_x = txt_pos_x - padding
    rect_pos_y = txt_pos_y + padding
    rect_position = (rect_pos_x, rect_pos_y)
    if color_bg is not None:
        rect_w = text_w + padding
        rect_h = text_h + padding
        x_start, x_stop = np.sort([rect_pos_x, txt_pos_x+rect_w])
        y_start, y_stop = np.sort([rect_pos_y, txt_pos_y-rect_h])
        crop = img[np.max([y_start, 0]):y_stop, np.max([x_start, 0]):x_stop]
        if not (x_start < 0 or x_stop < 0 or y_start < 0 or y_stop < 0 or crop.size == 0):
            bg = np.ones_like(crop) * np.array(color_bg, dtype=crop.dtype)
            img[np.max([y_start, 0]):y_stop, np.max([x_start, 0]):x_stop] = (
                cv2.addWeighted(crop, (1-alpha_bg), bg, alpha_bg, 0.0))

    cv2.putText(
        img,
        text,
        text_position,
        fontFace=fontFace,
        fontScale=fontScale,
        color=np.array(color_txt) * darken,
        thickness=thickness,
        lineType=lineType,
    )
    return text_size


def scale_lightness(rgb, scale_l=1.4):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def colored_body_parts_overlay(img, masks, clip=True, interpolation=cv2.INTER_CUBIC, alpha=0.28, mask_threshold=0.0, weight_scale=1, rgb=False):
    width, height = img.shape[1], img.shape[0]
    white_bckg = np.ones_like(img) * 255
    for i in range(masks.shape[0]):
        mask = cv2.resize(masks[i], dsize=(width, height), interpolation=interpolation)
        if clip:
            mask = np.clip(mask, 0, 1)
        else:
            mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(np.uint8)
        weight = mask
        mask_alpha = np.ones_like(weight)
        mask_alpha[mask < mask_threshold] = 0
        mask_alpha = np.expand_dims(mask_alpha, 2)
        weight = np.expand_dims(weight, 2) / weight_scale
        color_img = np.zeros_like(img)
        color = scale_lightness(cm.gist_rainbow(i / (len(masks)-1))[0:-1])
        color_img[:] = np.flip(np.array(color)*255).astype(np.uint8)
        white_bckg = white_bckg * (1 - mask_alpha * weight) + color_img * mask_alpha * weight
    heatmap = masks.sum(axis=0).clip(0, 1)
    heatmap = cv2.resize(heatmap, dsize=(width, height), interpolation=interpolation)
    mask_alpha = heatmap
    mask_alpha[heatmap < mask_threshold] = 0
    mask_alpha = np.repeat(np.expand_dims(mask_alpha, 2), 3, 2)
    white_bckg = white_bckg.astype(img.dtype)
    if rgb:
        white_bckg = cv2.cvtColor(white_bckg, cv2.COLOR_BGR2RGB)
    masked_img = (
            img * (1 - mask_alpha * alpha) + white_bckg * mask_alpha * alpha
    )
    return masked_img