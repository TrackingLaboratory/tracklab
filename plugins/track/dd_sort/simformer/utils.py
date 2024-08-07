import numpy as np
import torch

def convert_ltwh_to_ltrb(bboxes_ltwh):
    """
    Convert bounding boxes from LTWH format to LTRB format.

    Args:
    - bboxes_ltwh (numpy.ndarray or torch.Tensor): Bounding boxes in LTWH format of shape (B, N, 4),
                                  where B is the batch size, N is the number of bounding boxes,
                                  and the last dimension represents (left, top, width, height).

    Returns:
    - numpy.ndarray or torch.Tensor: Bounding boxes in LTRB format of shape (B, N, 4),
                    where the last dimension represents (left, top, right, bottom).
    """
    if isinstance(bboxes_ltwh, np.ndarray):
        # Convert NumPy array to PyTorch tensor
        bboxes_ltwh_tensor = torch.tensor(bboxes_ltwh, dtype=torch.float32)
    elif isinstance(bboxes_ltwh, torch.Tensor):
        bboxes_ltwh_tensor = bboxes_ltwh
    else:
        raise ValueError("Unsupported input type. Input should be a NumPy array or a PyTorch tensor.")

    # Extract the left, top, width, and height components
    left, top, width, height = torch.split(bboxes_ltwh_tensor, 1, dim=-1)

    # Calculate right and bottom components
    right = left + width
    bottom = top + height

    # Concatenate the components to form LTRB bounding boxes
    bboxes_ltrb_tensor = torch.cat((left, top, right, bottom), dim=-1)

    # Convert the result back to NumPy array if the input was a NumPy array
    if isinstance(bboxes_ltwh, np.ndarray):
        bboxes_ltrb_numpy = bboxes_ltrb_tensor.numpy()
        return bboxes_ltrb_numpy
    else:
        return bboxes_ltrb_tensor


def bbox_ltwh2ltrb(ltwh):
    return np.concatenate((ltwh[:2], ltwh[:2] + ltwh[2:]))


def unnormalize_bbox(bbox, image_shape):
    return bbox * (list(image_shape) * 2)


def normalize_bbox(bbox, image_shape):
    return bbox / image_shape.repeat(2)


def normalize_kps(kps, image_shape):
    nm_kps = kps.clone()
    nm_kps[..., :2] = kps[..., :2] / image_shape
    return nm_kps
