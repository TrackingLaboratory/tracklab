# vim: expandtab:ts=4:sw=4
import numpy as np
import torch
try:
    from torchreid.metrics import compute_distance_matrix
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
except ImportError:
    torchreid = None

from torch.nn import functional as F

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
    """
    x_ = torch.from_numpy(np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True))
    y_ = torch.from_numpy(np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True))
    distances = compute_distance_matrix(x_, y_, metric='euclidean')
    return np.maximum(0.0, torch.min(distances, axis=0)[0].numpy())


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.
    """
    x_ = torch.from_numpy(np.asarray(x))
    y_ = torch.from_numpy(np.asarray(y))
    distances = compute_distance_matrix(x_, y_, metric='cosine')
    distances = distances.cpu().detach().numpy()
    return distances.min(axis=0)


def _nn_part_based(y, x, normalize_features=True):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    x_features = torch.from_numpy(x['reid_features'])
    x_visibility_scores = torch.from_numpy(x['visibility_scores'])
    y_features = torch.from_numpy(y[-1]['reid_features'])
    y_visibility_scores = torch.from_numpy(y[-1]['visibility_scores'])
    if normalize_features:
        """Features are normalized here, which is not the most efficient since a given tracklet features will be normalized multiple times.
        It is not done before because we want to do the exponential moving average (EMA) aggregation of tracklets features on the un-normalized features
        and then normalize after. If normalization was done before EMA, the averaged features would not be normalized 
        anymore and we would therefore pass un-normalized features to 'compute_distance_matrix_using_bp_features'."""
        x_features = F.normalize(x_features, p=2, dim=-1)
        y_features = F.normalize(y_features, p=2, dim=-1)
    distances = compute_distance_matrix_using_bp_features(y_features.unsqueeze(0),
                                                          x_features,
                                                          y_visibility_scores.unsqueeze(0),
                                                          x_visibility_scores,
                                                          use_gpu=False,
                                                          )
    distances = distances[0] / 2    # When feature are normalized, the above function returns distances within [0, 2]
    # TODO handle invalid distances
    return distances.mean(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        elif metric == "part_based":
            self._metric = _nn_part_based
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.
        active_targets = confirmed tracks
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of len N with associated target identities, i.e. track ids
        active_targets : List[int]
            A list of targets that are currently present in the scene, i.e. confirmed tracks.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)  # in our case, only one feature per track
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]  # for each track, keep only the last 'budget' reid features
        self.samples = {k: self.samples[k] for k in active_targets}  # keep only confirmed tracks

    def distance(self, features, targets):
        """Compute distance between features and targets.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        cost_matrix = np.zeros((len(targets), len(features['reid_features'])))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
