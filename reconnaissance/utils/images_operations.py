import cv2
import numpy as np


def overlay_heatmap(img, heatmap):
    width, height = img.shape[1], img.shape[0]
    heatmap = cv2.resize(heatmap, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_gray = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    # heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    img_with_heatmap = cv2.addWeighted(img, 0.5, heatmap_color.astype(img.dtype), 0.5, 0)
    # mask_threshold = 0.005
    # mask_alpha = np.ones_like(mask)
    # mask_alpha[mask_alpha < mask_threshold] = 0
    # mask_alpha = np.repeat(np.expand_dims(mask, 2), 3, 2)
    # masked_img = img_crop * (1-mask_alpha*0.5) + mask_color.astype(img_crop.dtype) * mask_alpha*0.5
    return img_with_heatmap