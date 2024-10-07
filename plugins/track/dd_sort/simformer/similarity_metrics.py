# random_assignement
# euclidean_sim_matrix
# iou_sim_matrix
import torch
import torch.nn.functional as F

from .utils import convert_ltwh_to_ltrb


def cosine_sim_matrix(track_embs, track_masks, det_embs, det_masks):
    """
    Compute the cosine similarity between the tokens of dets and tracks.
    tracks are on dim = 1 and dets on dim = 2
    masked pairs are set to -inf
    """
    td_sim_matrix = F.cosine_similarity(track_embs.unsqueeze(2), det_embs.unsqueeze(1), dim=3)
    td_sim_matrix = (td_sim_matrix + 1) / 2
    td_sim_matrix[~(track_masks.unsqueeze(2) * det_masks.unsqueeze(1))] = -float("inf")
    return td_sim_matrix


def random_sim_matrix(track_embs, track_masks, det_embs, det_masks):
    """
    track_embs: Tensor [B, T(+P), E]
    track_masks: Tensor [B, T(+P)]
    det_embs: Tensor [B, D(+P), E]
    det_masks: Tensor [B, D(+P)]

    returns:
    td_sim_matrix: Tensor [B, T(+P), D(+P)]
        padded pairs are set to -inf
    """
    td_sim_matrix = torch.rand((det_embs.shape[0], track_embs.shape[1], det_embs.shape[1]), device=det_embs.device)
    td_sim_matrix[~(track_masks.unsqueeze(2) * det_masks.unsqueeze(1))] = -float("inf")
    return td_sim_matrix


def euclidean_sim_matrix(track_embs, track_masks, det_embs, det_masks):
    """
    track_embs: Tensor [B, T(+P), E]
    track_masks: Tensor [B, T(+P)]
    det_embs: Tensor [B, D(+P), E]
    det_masks: Tensor [B, D(+P)]

    returns:
    td_sim_matrix: Tensor [B, T(+P), D(+P)]
        padded pairs are set to -inf
    """
    td_sim_matrix = -torch.cdist(track_embs, det_embs, p=2)
    td_sim_matrix[~(track_masks.unsqueeze(2) * det_masks.unsqueeze(1))] = -float("inf")
    return td_sim_matrix


def norm_euclidean_sim_matrix(track_embs, track_masks, det_embs, det_masks):
    """
    track_embs: Tensor [B, T(+P), E]
    track_masks: Tensor [B, T(+P)]
    det_embs: Tensor [B, D(+P), E]
    det_masks: Tensor [B, D(+P)]

    returns:
    td_sim_matrix: Tensor [B, T(+P), D(+P)]
        padded pairs are set to -inf
    """
    track_embs = F.normalize(track_embs, p=2, dim=-1)
    det_embs = F.normalize(det_embs, p=2, dim=-1)
    td_sim_matrix = torch.cdist(track_embs, det_embs, p=2)
    td_sim_matrix = 1 - td_sim_matrix/2
    td_sim_matrix[~(track_masks.unsqueeze(2) * det_masks.unsqueeze(1))] = -float("inf")
    return td_sim_matrix


def iou_sim_matrix(track_bboxes, track_masks, det_bboxes, det_masks):
    """
    Computes batched IoU between two sets of bounding boxes.
    Args:
    - boxA, boxB: tensors of shape (B, N, 4) and (B, M, 4) respectively.
                  Each slice along dim 1 is a set of bounding boxes,
                  represented as [left, top, right, bottom].
    Returns:
    - IoU matrix of shape (B, N, M).
    """

    track_bboxes = convert_ltwh_to_ltrb(track_bboxes)
    det_bboxes = convert_ltwh_to_ltrb(det_bboxes)

    B, N, _ = track_bboxes.shape
    _, M, _ = det_bboxes.shape

    # Expand dimensions for broadcasting
    boxA_exp = track_bboxes.unsqueeze(2).expand(B, N, M, 4)
    boxB_exp = det_bboxes.unsqueeze(1).expand(B, N, M, 4)

    # Compute the (x, y)-coordinates of the intersection rectangles
    left = torch.max(boxA_exp[..., 0], boxB_exp[..., 0])
    top = torch.max(boxA_exp[..., 1], boxB_exp[..., 1])
    right = torch.min(boxA_exp[..., 2], boxB_exp[..., 2])
    bottom = torch.min(boxA_exp[..., 3], boxB_exp[..., 3])

    # Compute the area of intersection rectangles
    inter_area = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)

    # Compute the area of both input bounding boxes
    boxA_area = (boxA_exp[..., 2] - boxA_exp[..., 0]) * (boxA_exp[..., 3] - boxA_exp[..., 1])
    boxB_area = (boxB_exp[..., 2] - boxB_exp[..., 0]) * (boxB_exp[..., 3] - boxB_exp[..., 1])

    # Compute the IoU
    iou_td_sim_matrix = inter_area / (boxA_area + boxB_area - inter_area)

    # Remove entries for non-existing bboxes
    iou_td_sim_matrix[~(track_masks.unsqueeze(2) * det_masks.unsqueeze(1))] = -float("inf")

    return iou_td_sim_matrix


similarity_metrics = {
    "cosine": cosine_sim_matrix,
    "euclidean": euclidean_sim_matrix,
    "norm_euclidean": norm_euclidean_sim_matrix,
    "iou": iou_sim_matrix,
    "random": random_sim_matrix,
    "default_for_each_token_type": None,  # a specific similarity metric among above ones will be chosen for each token type
}
