import cv2
import numpy as np


def overlay_heatmap(img, heatmap, weight=0.5, mask_threshold=0.0, color_map=cv2.COLORMAP_JET):
    """
    Overlay a heatmap on an image with given color map.
    Args:
        img: input image in OpenCV BGR format
        heatmap: heatmap to overlay in float from range 0->1. Will be resized to image size and values clipped to 0->1.
        weight: alpha blending weight between image and heatmap, 1-weight for image and weight for heatmap. Set to -1 to
        use the heatmap as alpha channel.
        color_map: OpenCV type colormap to use for heatmap.
        mask_threshold: heatmap values below this threshold will not be displayed.

    Returns:
        Image with heatmap overlayed.
    """
    width, height = img.shape[1], img.shape[0]
    heatmap = cv2.resize(heatmap, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_gray = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, color_map).astype(img.dtype)
    mask_alpha = np.ones_like(heatmap)
    mask_alpha[heatmap < mask_threshold] = 0
    mask_alpha = np.repeat(np.expand_dims(mask_alpha, 2), 3, 2)
    if weight == -1:
        weight = np.repeat(np.expand_dims(heatmap/2, 2), 3, 2)
    img_with_heatmap = img * (1-mask_alpha*weight) + heatmap_color * mask_alpha*weight
    # img_with_heatmap = cv2.addWeighted(img, 1-weight, heatmap_color.astype(img.dtype), weight, 0)
    return img_with_heatmap