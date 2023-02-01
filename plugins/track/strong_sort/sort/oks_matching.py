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
    # TODO should be replaced by a mean keypoints and not last detected one
    # we won't be able to compute confidences so we will have to check if (x, y) != 0
    keypoints_bool = keypoints[:, 2] > 0.0
    tl, br = np.amax(keypoints[keypoints_bool], axis=0), np.amin(
        keypoints[keypoints_bool], axis=0
    )
    scale = np.sqrt((br[0] - tl[0]) ** 2 + (br[1] - tl[1]) ** 2)
    distances = np.sqrt(
        (keypoints[:, 0] - candidates[:, :, 0]) ** 2
        + (keypoints[:, 1] - candidates[:, :, 1]) ** 2
    )
    distances = np.exp(
        -(distances**2) / (2 * scale**2 * kappa**2)
    ) * keypoints_bool.astype(float)
    oks = np.sum(distances, axis=1) / keypoints_bool.sum()
    return oks


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
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        # TODO modify here to use mean keypoints
        keypoints = tracks[track_idx].last_detection_keypoints()
        candidates = np.asarray([detections[i].keypoints for i in detection_indices])
        cost_matrix[row, :] = 1.0 - oks(keypoints, candidates)
    return cost_matrix
