import numpy as np


def kp_to_bbox(kp_xy):
    """Extract bounding box from keypoints.
    Args:
        kp_xy (np.ndarray): keypoints in image coordinates, shape (K, 2)
    Returns:
        bbox (np.ndarray): bounding box ltwh (COCO format), shape (4,)
    """
    lt = np.amin(kp_xy, axis=0)
    br = np.amax(kp_xy, axis=0)
    w = br[0] - lt[0]
    h = br[1] - lt[1]
    return np.array([lt[0], lt[1], w, h])


def openpifpaf_kp_to_bbox(kp_xy):
    """Extract bounding box from keypoints for openpifpaf framework.
    Args:
        kp_xy (np.ndarray): keypoints in image coordinates, shape (K, 2)
    Returns:
        bbox (np.ndarray): bounding box ltwh (COCO format), shape (4,)
    """
    # remove keypoints that have not been found
    kp_xy = kp_xy[kp_xy[:, 2] > 0]
    lt = np.amin(kp_xy, axis=0)
    br = np.amax(kp_xy, axis=0)
    w = br[0] - lt[0]
    h = br[1] - lt[1]
    return np.array([lt[0], lt[1], w, h])


def kp_to_bbox_w_threshold(kp_xyc, vis_threshold=0.0):
    """Extract bounding box from keypoints with visibility threshold.
    Args:
        kp_xy (np.ndarray): keypoints in image coordinates, shape (K, 3) (x, y, c)
    Returns:
        bbox (np.ndarray): bounding box ltwh (COCO format), shape (4,)
    """
    kp_xy = kp_xyc[kp_xyc[:, 2] > vis_threshold]
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
    kp_xyc_bbox[
        (kp_xyc_bbox[:, 2] == 0)
        | (kp_xyc_bbox[:, 0] < 0)
        | (kp_xyc_bbox[:, 0] >= w)
        | (kp_xyc_bbox[:, 1] < 0)
        | (kp_xyc_bbox[:, 1] >= h)
    ] = 0

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


def clip_bbox_ltwh_to_img_dim(bbox_ltwh, img_w, img_h):
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
    l = np.clip(l, 0, img_w - 1)
    t = np.clip(t, 0, img_h - 1)
    w = np.clip(w, 0, img_w - 1 - l)
    h = np.clip(h, 0, img_h - 1 - t)
    assert np.equal(
        np.array([l, t, w, h]), clip_bbox_ltwh_to_img_dim_old(bbox_ltwh, img_w, img_h)
    ).all()
    return np.array([l, t, w, h])


def clip_bbox_ltwh_to_img_dim_old(bbox_ltwh, img_w, img_h):
    """
    Clip bounding box to image dimensions.
    Args:
        bbox_ltwh (np.ndarray): bounding box, shape (4,)
        img_w (int): image width
        img_h (int): image height

    Returns:
        bbox_ltwh (np.ndarray): clipped bounding box, shape (4,)
    """
    l, t, w, h = bbox_ltwh.copy()
    l = max(l, 0)
    t = max(t, 0)
    w = min(l + w, img_w - 1) - l
    h = min(t + h, img_h - 1) - t
    return np.array([l, t, w, h])


def clip_bbox_ltrb_to_img_dim(bbox_ltrb, img_w, img_h):
    """
    Clip bounding box to image dimensions.
    Args:
        bbox_ltrb (np.ndarray): bounding box, shape (4,)
        img_w (int): image width
        img_h (int): image height
    Returns:
        bbox_ltrb (np.ndarray): clipped bounding box, shape (4,)
    """
    l, t, r, b = bbox_ltrb
    l = np.clip(l, 0, img_w - 1)
    t = np.clip(t, 0, img_h - 1)
    r = np.clip(r, 0, img_w - 1)
    b = np.clip(b, 0, img_h - 1)
    return np.array([l, t, r, b])


def round_bbox_coordinates(bbox):
    """
    Round bounding box coordinates.
    Round to ceil value to avoid bbox with zero width or height.
    Because of ceil rounding, resulting bbox may be outside of image.
    Apply above 'clip_bbox_ltrb_to_img_dim' to clip bbox to image dimensions.
    Args:
        bbox (np.ndarray): bounding box, shape (4,), in tlwh or tlbr format
    Returns:
        bbox (np.ndarray): rounded bounding box, shape (4,)
    """

    return np.concatenate([np.floor(bbox[:2]), np.ceil(bbox[2:])]).astype(int)


def bbox_ltwh2ltrb(ltwh):
    return np.concatenate((ltwh[:2], ltwh[:2] + ltwh[2:]))


def bbox_ltrb2ltwh(ltrb):
    return np.concatenate((ltrb[:2], ltrb[2:] - ltrb[:2]))


def bbox_ltwh2cmwh(ltwh):
    return np.concatenate((ltwh[:2] + ltwh[2:] / 2, ltwh[2:]))


def generate_bbox_from_keypoints(keypoints, extension_factor, image_shape):
    """
    Generates a bounding box from keypoints by computing the bounding box of the keypoints and extending it by a factor.

    Args:
        keypoints (np.ndarray): A numpy array of shape (K, 3) representing the keypoints in the format (x, y, c).
        extension_factor (tuple): A tuple of float [top, bottom, right&left] representing the factor by which
        the bounding box should be extended based on the keypoints.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the bounding box in the format (left, top, w, h).
    """
    keypoints = sanitize_keypoints(keypoints, image_shape)
    lt, rb = np.min(keypoints[:, :2], axis=0), np.max(keypoints[:, :2], axis=0)
    w, h = rb - lt
    lt -= np.array([extension_factor[2]*w, extension_factor[0]*h])
    rb += np.array([extension_factor[2]*w, extension_factor[1]*h])
    bbox = np.concatenate([lt, rb - lt])
    bbox = sanitize_bbox_ltwh(bbox, image_shape)
    return bbox


def sanitize_keypoints(keypoints, image_shape):
    """
    Sanitizes keypoints by clipping them to the image dimensions and ensuring that their confidence values are valid.

    Args:
        keypoints (np.ndarray): A numpy array of shape (K, 3) representing the keypoints in the format (x, y, c).
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.

    Returns:
        np.ndarray: A numpy array of shape (K, 3) representing the sanitized keypoints in the format (x, y, c).
    """
    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_shape[0] - 1)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_shape[1] - 1)
    return keypoints


def sanitize_bbox_ltwh(bbox, image_shape):
    """
    Sanitizes a bounding box by clipping it to the image dimensions and ensuring that its dimensions are valid.

    Args:
        box (np.ndarray): A numpy array of shape (4,) representing the bounding box in the format
        `[left, top, width, height]`.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the sanitized bounding box in the format
        `[left, top, width, height]`.
    """
    bbox[0] = max(0, min(bbox[0], image_shape[0] - 2))
    bbox[1] = max(0, min(bbox[1], image_shape[1] - 2))
    bbox[2] = max(1, min(bbox[2], image_shape[0] - 1 - bbox[0]))
    bbox[3] = max(1, min(bbox[3], image_shape[1] - 1 - bbox[1]))
    return np.array(bbox)


def ltwh_to_xywh(bbox, image_shape=None):
    """
    Converts coordinates `[left, top, w, h]` to `[center_x, center_y, w, h]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    if image_shape:
        bbox = sanitize_bbox_ltwh(bbox, image_shape)
    return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]])


def ltwh_to_ltrb(bbox, image_shape=None):
    """
    Converts coordinates `[left, top, w, h]` to `[left, top, right, bottom]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    if image_shape:
        bbox = sanitize_bbox_ltwh(bbox, image_shape)
    return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])


def sanitize_bbox_ltrb(bbox, image_shape):
    """
    Sanitizes a bounding box by clipping it to the image dimensions and ensuring that its dimensions are valid.

    Args:
        box (np.ndarray): A numpy array of shape (4,) representing the bounding box in the format
        `[left, top, right, bottom]`.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the sanitized bounding box in the format
        `[left, top, right, bottom]`.
    """
    bbox[0] = max(0, min(bbox[0], image_shape[0] - 2))  # ensure width > 0
    bbox[1] = max(0, min(bbox[1], image_shape[1] - 2))  # ensure height > 0
    bbox[2] = max(1, min(bbox[2], image_shape[0] - 1))  # ensure width > 0
    bbox[3] = max(1, min(bbox[3], image_shape[1] - 1))  # ensure height > 0
    return np.array(bbox)


def ltrb_to_xywh(bbox, image_shape=None):
    """
    Converts coordinates `[left, top, right, bottom]` to `[center_x, center_y, w, h]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    if image_shape:
        bbox = sanitize_bbox_ltrb(bbox, image_shape)
    return np.array(
        [
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
            (bbox[2] - bbox[0]),
            (bbox[3] - bbox[1]),
        ]
    )


def ltrb_to_ltwh(bbox, image_shape=None):
    """
    Converts coordinates `[left, top, right, bottom]` to `[left, top, w, h]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    if image_shape:
        bbox = sanitize_bbox_ltrb(bbox, image_shape)
    return np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
