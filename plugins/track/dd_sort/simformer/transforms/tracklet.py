import logging

import numpy as np

from .transform import Transform

log = logging.getLogger(__name__)


class RandomGapsTracklet(Transform):
    """
    Add some gaps in the tracklet.
    """
    def __init__(self,
                 max_gap_size: int = 1,
                 max_gaps: int = 1,
                 ):
        super().__init__()
        self.max_gap_size = max_gap_size
        self.max_gaps = max_gaps

    def __call__(self, df):
        gaps = self.rng.integers(self.max_gaps, endpoint=True)
        gaps = min(gaps, len(df) // 2)
        for _ in range(gaps):
            start_idx = self.rng.integers(len(df))
            length = self.rng.integers(1, self.max_gap_size, endpoint=True)
            end_idx = min(len(df)-1, start_idx+length)
            df = df.drop(df.index[start_idx:end_idx])
        return df

class RandomObsGapTracklet(Transform):
    """
    Keep only detections that are older than a random value.
    """
    def __init__(self, std_age: int = 5):
        super().__init__()
        self.std_age = std_age

    def __call__(self, df):
        cutoff_age = int(np.abs(self.rng.normal(0, self.std_age)))
        age = df.image_id.max() - df.image_id
        df = df[(age >= cutoff_age) | (df.to_match != 0)]
        return df


class RandomLengthTracklet(Transform):
    """
    Keep only the last n detections of a random length n.
    """
    def __init__(self, max_length: int = 10):
        super().__init__()
        self.max_length = max_length

    def __call__(self, df):
        length = int(self.rng.uniform(1, self.max_length))
        cutoff = len(df) - (length + 1)
        df = df.iloc[cutoff:]
        return df