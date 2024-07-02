import numpy as np

from .transform import Transform


class RandomBboxShiftScale(Transform):
    def __init__(self, shift_std=1.0, scale_limit=0.1):
        super().__init__()
        self.shift_std = shift_std
        self.scale_limit = scale_limit

    def __call__(self, df):
        assert "bbox_ltwh" in df.columns
        assert "keypoints_xyc" in df.columns
        bbox_shift = self.rng.normal(scale=self.shift_std, size=2)
        bbox_scale = self.rng.uniform(1-self.scale_limit, 1+self.scale_limit, size=2)
        df.bbox_ltwh[:, :2] += bbox_shift
        df.bbox_ltwh[:, 2:] *= bbox_scale
        kp_xyc = df.keypoints_xyc
        kp_xyc[
            (kp_xyc[:, 2] == 0)
            | (kp_xyc[:, 0] < 0)
            | (kp_xyc[:, 0] >= df.bbox_ltwh[:,])
            | (kp_xyc[:, 1] < 0)
            | (kp_xyc[:, 1] >= df.bbox)
            ] = 0
        return df
