import numpy as np
import logging

from .transform import Transform

log = logging.getLogger(__name__)


class MaxTrackletObs(Transform):
    """
    Limit the number of observations in the tracklet.
    """

    def __init__(self, max_obs: int = 200):
        super().__init__()
        self.max_obs = max_obs
        assert self.max_obs > 0, "'max_obs' must be greater than 0."

    def __call__(self, df):
        return df.tail(self.max_obs)


class SporadicTrackletDropout(Transform):
    """
    Randomly drop some detections from the tracklet.
    """

    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.p_drop = p_drop
        assert 0 <= self.p_drop <= 1, "'p_drop' must be in the range [0, 1]."

    def __call__(self, df):
        mask = self.rng.uniform(size=len(df)) > self.p_drop
        if mask.sum() == 0:
            mask[-1] = True
        return df[mask]


class StructuredTrackletDropout(Transform):
    """
    This function randomly proposes windows of detections to drop from the tracklet.
    It stores these proposed windows in a buffer and then randomly selects up to
    max_num_windows from the buffer to drop. If max_num_windows is set to -1,
    all proposed windows are dropped.

    - p_drop: Probability of proposing a window for dropping.
    - max_drop: Maximum number of detections to drop per window.
    - max_num_windows: Maximum number of windows to drop. If -1, no limit is applied.
    """

    def __init__(self, p_drop: float = 0.1, max_drop: int = 5, max_num_windows: int = 2):
        super().__init__()
        self.p_drop = p_drop
        assert 0 <= self.p_drop <= 1, "'p_drop' must be in the range [0, 1]."
        self.max_drop = max_drop
        self.max_num_windows = max_num_windows

    def __call__(self, df):
        drop_proposals = []

        indices = df.index.tolist()

        i = 0
        while i < len(indices):
            if self.rng.uniform() < self.p_drop:
                drop_length = self.rng.integers(1, self.max_drop + 1)
                drop_proposals.append(indices[i:i + drop_length])
                i += drop_length
            else:
                i += 1

        if self.max_num_windows != -1:
            selected_drops_idx = np.sort(
                self.rng.choice(len(drop_proposals), size=min(self.max_num_windows, len(drop_proposals)),
                                replace=False))
            selected_drops = [drop_proposals[idx] for idx in selected_drops_idx]
        else:
            selected_drops = drop_proposals

        drop_indices = [idx for drop_range in selected_drops for idx in drop_range]
        if len(drop_indices) >= len(df):
            drop_indices = drop_indices[:-1]
        return df.drop(drop_indices)
