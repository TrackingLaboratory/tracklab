from __future__ import absolute_import
import numpy as np
from . import linear_assignment

# per keypoint that controls the falloff
# (maybe remove the ears ?)
kappa = np.array(
    [
        0.026,  # nose
        0.025,  # eyes
        0.025,  # eyes
        0.035,  # ears
        0.035,  # ears
        0.079,  # shoulders
        0.079,  # shoulders
        0.072,  # elbows
        0.072,  # elbows
        0.062,  # wrists
        0.062,  # wrists
        0.107,  # hips
        0.107,  # hips
        0.087,  # knees
        0.087,  # knees
        0.089,  # ankles
        0.089,  # ankles
    ]
)


def oks(keypoints, candidates):
    """Computer object keypoint similarity.

    Parameters
    ----------
    keypoints : ndarray (N, 3) (x, y, conf)
    candidates : ndarray (M, N, 3) (M, x, y, conf)
        A matrix of candidate keypoints in the same format as `keypoints`.

    Returns
    -------
    ndarray
        The object keypoint similarity in [0, 1] between the `keypoints` and each
        candidate. A higher score means a beter similarity between the keypoints.
    """
    visible = keypoints[:, 2] > 0.0

    tl, br = np.amin(keypoints[visible], axis=0), np.amax(keypoints[visible], axis=0)
    total_tl, total_br = np.amin(keypoints, axis=0), np.amax(keypoints, axis=0)

    area = (br[0] - tl[0]) * (br[1] - tl[1])  # using visible keypoints
    total_area = (total_br[0] - total_tl[0]) * (
        total_br[1] - total_tl[1]
    )  # using all keypoints

    # Compute the rotation of the skeleton (prevent from aligned keypoints -> area = 0)
    c, s = np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))
    rotate = np.array(((c, -s), (s, c)))
    keypoints_45 = np.copy(keypoints)
    keypoints_45[:, :2] = np.einsum("ij,kj->ki", rotate, keypoints_45[:, :2])

    tl_45, br_45 = np.amin(keypoints_45[visible], axis=0), np.amax(
        keypoints_45[visible], axis=0
    )  # using visible rotated keypoints
    total_tl_45, total_br_45 = np.amin(keypoints_45, axis=0), np.amax(
        keypoints_45, axis=0
    )  # using all rotated keypoints

    area_45 = (br_45[0] - tl_45[0]) * (br_45[1] - tl_45[1])  # w * h
    total_area_45 = (total_br_45[0] - total_tl_45[0]) * (
        total_br_45[1] - total_tl_45[1]
    )  # w * h

    factor = np.sqrt(
        min(
            total_area / area if area > 0.1 else np.inf,
            total_area_45 / area_45 if area_45 > 0.1 else np.inf,
        )
    )

    factor_clipped = min(5.0, factor)
    scale = np.sqrt(area) * factor_clipped
    if scale < 0.1:
        scale = np.nan

    distances = np.sqrt(
        (keypoints[:, 0] - candidates[:, :, 0]) ** 2
        + (keypoints[:, 1] - candidates[:, :, 1]) ** 2
    )
    oks_by_keypoint = np.exp(
        -(distances**2) / (2 * scale**2 * kappa**2)
    ) * visible.astype(float)
    return np.sum(oks_by_keypoint, axis=1) / visible.sum()


def oks_cost(tracks, detections, track_indices=None, detection_indices=None):
    """An object keypoints similarity distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - oks(tracks[track_indices[i]], detections[detection_indices[j]])`.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        keypoints = tracks[track_idx].last_detection.keypoints
        candidates = np.asarray([detections[i].keypoints for i in detection_indices])
        cost_matrix[row, :] = 1.0 - oks(keypoints, candidates)
    return cost_matrix
