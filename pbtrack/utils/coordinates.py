import numpy as np

def kp_to_bbox(kp_xy):
    """Extract bounding box from keypoints.
    Args:
        kp_xy (np.ndarray): keypoints in image coordinates, shape (K, 2)
    Returns:
        bbox (np.ndarray): bounding box tlwh (COCO format), shape (4,)
    """
    lt = np.amin(kp_xy, axis=0)
    br = np.amax(kp_xy, axis=0)
    w = br[0] - lt[0]
    h = br[1] - lt[1]
    return np.array([lt[0], lt[1], w, h])

def kp_to_bbox_w_threshold(kp_xyc, vis_threshold=0.1):
    """Extract bounding box from keypoints with visibility threshold.
    Args:
        kp_xy (np.ndarray): keypoints in image coordinates, shape (K, 3) (x, y, c)
    Returns:
        bbox (np.ndarray): bounding box tlwh (COCO format), shape (4,)
    """
    kp_xy = kp_xyc[kp_xyc[:, 2] > vis_threshold][:, :2]
    return kp_to_bbox(kp_xy)

def kp_img_to_kp_bbox(kp_xyc_img, bbox_ltwh):
    """
    Convert keypoints in image coordinates to bounding box coordinates and filter out keypoints that are outside the
    bounding box.
    Args:
        kp_xyc_img (np.ndarray): keypoints in image coordinates, shape (K, 2)
        bbox_tlwh (np.ndarray): bounding box, shape (4,)
    Returns:
        kp_xyc_bbox (np.ndarray): keypoints in bounding box coordinates, shape (K, 2)
    """
    l, t, w, h = bbox_ltwh
    kp_xyc_bbox = kp_xyc_img.copy()

    # put keypoints in bbox coord space
    kp_xyc_bbox[:, 0] = kp_xyc_img[:, 0] - l
    kp_xyc_bbox[:, 1] = kp_xyc_img[:, 1] - t

    # remove out of bbox keypoints
    kp_xyc_bbox[(kp_xyc_bbox[:, 2] == 0)
                | (kp_xyc_bbox[:, 0] < 0)
                | (kp_xyc_bbox[:, 0] >= w)
                | (kp_xyc_bbox[:, 1] < 0)
                | (kp_xyc_bbox[:, 1] >= h)] = 0

    return kp_xyc_bbox


def rescale_keypoints(rf_keypoints, size, new_size):
    """
    Rescale keypoints to new size.
    Args:
        rf_keypoints (np.ndarray): keypoints in relative coordinates, shape (K, 2)
        size (tuple): original size, (w, h)
        new_size (tuple): new size, (w, h)
    Returns:
        rf_keypoints (np.ndarray): rescaled keypoints in relative coordinates, shape (K, 2)
    """
    w, h = size
    new_w, new_h = new_size
    rf_keypoints[:, 0] = rf_keypoints[:, 0] * new_w / w
    rf_keypoints[:, 1] = rf_keypoints[:, 1] * new_h / h
    return rf_keypoints


def clip_to_img_dim(bbox_ltwh, img_w, img_h):
    """
    Clip bounding box to image dimensions.
    Args:
        bbox_ltwh (np.ndarray): bounding box, shape (4,)
        img_w (int): image width
        img_h (int): image height
    Returns:
        bbox_ltwh (np.ndarray): clipped bounding box, shape (4,)
    """
    l, t, w, h = bbox_ltwh
    l = np.clip(l, 0, img_w)
    t = np.clip(t, 0, img_h)
    w = np.clip(w, 0, img_w - l)
    h = np.clip(h, 0, img_h - t)
    assert np.equal(np.array([l, t, w, h]), clip_to_img_dim_old(bbox_ltwh, img_w, img_h)).all()
    return np.array([l, t, w, h])


def clip_to_img_dim_old(bbox_ltwh, img_w, img_h):
    """
    Clip bounding box to image dimensions.
    Args:
        bbox_ltwh:
        img_w:
        img_h:

    Returns:

    """
    l, t, w, h = bbox_ltwh.copy()
    l = max(l, 0)
    t = max(t, 0)
    w = min(l+w, img_w) - l
    h = min(t+h, img_h) - t
    return np.array([l, t, w, h])
