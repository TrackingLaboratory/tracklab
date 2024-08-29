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

    def __call__(self, df, video_df):
        return df.tail(self.max_obs)


class SporadicTrackletDropout(Transform):
    """
    Randomly drop some detections from the tracklet.
    """

    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.p_drop = p_drop
        assert 0 <= self.p_drop <= 1, "'p_drop' must be in the range [0, 1]."

    def __call__(self, df, video_df):
        if df.empty:
            return df
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

    def __call__(self, df, video_df):
        if df.empty:
            return df
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


class SwapRandomDetections(Transform):
    """
    Randomly swap two detections from the tracklet, respecting previous augmentations.
    """

    def __init__(self, p_swap: float = 0.1, max_swap_prop: float = 0.3):
        super().__init__()
        self.p_swap = p_swap
        self.max_swap_prop = max_swap_prop
        assert 0 <= self.p_swap <= 1.0, "'p_swap' must be in the range [0, 1.0]."
        assert 0 <= self.max_swap_prop <= 1.0, "'max_swap_prop' must be in the range [0, 1.0]."

    def __call__(self, df, video_df):
        if df.empty:
            return df

        # Ensure 'da_swapped' column exists
        if 'da_swapped' not in df.columns:
            df['da_swapped'] = False
        track_id = df['track_id'].unique()[0]

        # Generate a mask for swapping, excluding already swapped detections
        swap_mask = (self.rng.uniform(size=len(df)) < self.p_swap) & ~df['da_swapped']

        # Ensure that no more than max_swap_prop of the detections are swapped
        if swap_mask.sum() / len(df) > self.max_swap_prop:
            swap_idx = self.rng.choice(swap_mask.index[swap_mask], size=int(self.max_swap_prop * len(df)),
                                       replace=False)
            swap_mask = swap_mask.index.isin(swap_idx)

        # For the swap_mask that are True, we randomly select another detection to swap from the same 'image_id' in the 'video_df' if available
        swap_df = df[swap_mask].copy()
        for idx, row in swap_df.iterrows():
            image_id = row['image_id']
            swap_candidates = video_df[(video_df['image_id'] == image_id) & (video_df['track_id'] != track_id)].index
            if len(swap_candidates) > 1:  # Ensure there is at least one other eligible detection to swap with
                swap_idx = self.rng.choice(swap_candidates[swap_candidates != idx])
                swap_df.loc[idx] = video_df.loc[swap_idx]

        # Update the original dataframe with the swapped values
        df.loc[swap_mask] = swap_df

        # Mark the newly swapped detections
        df.loc[swap_mask, 'da_swapped'] = True
        df.loc[swap_mask, 'track_id'] = df.loc[~swap_mask, 'track_id'].unique()[0]
        return df


class SwapOccludedDetections(Transform):
    def __init__(self, max_swap_prop: float = 0.3, min_occlusion_count: int = 1):
        super().__init__()
        self.max_swap_prop = max_swap_prop
        self.min_occlusion_count = min_occlusion_count
        assert 0 <= self.max_swap_prop <= 1.0, "'max_swap_prop' must be in the range [0, 1.0]."
        assert self.min_occlusion_count >= 0, "'min_occlusion_count' must be non-negative."

    def __call__(self, df, video_df):
        if df.empty:
            return df

        # Ensure 'da_swapped' column exists
        if 'da_swapped' not in df.columns:
            df['da_swapped'] = False
        original_track_id = df['track_id'].unique()[0]

        # Identify eligible occluded detections in df (not previously swapped)
        eligible_mask_df = (df['occlusion_count'] >= self.min_occlusion_count) & (~df['da_swapped'])

        # Ensure that no more than max_swap_prop of the detections are swapped
        if eligible_mask_df.sum() / len(df) > self.max_swap_prop:
            swap_idx = self.rng.choice(eligible_mask_df.index[eligible_mask_df],
                                       size=int(self.max_swap_prop * len(df)),
                                       replace=False)
            eligible_mask_df = eligible_mask_df.index.isin(swap_idx)

        # Iterate through eligible detections in df
        for idx in df[eligible_mask_df].index:
            image_id = df.loc[idx, 'image_id']

            # Find eligible detections in video_df for the same image_id
            eligible_mask_video = (
                    (video_df['image_id'] == image_id) &
                    (video_df['occlusion_count'] >= self.min_occlusion_count) &
                    (video_df['track_id'] != original_track_id)
            )

            if eligible_mask_video.any():
                # Randomly select one eligible detection from video_df
                swap_idx = self.rng.choice(video_df[eligible_mask_video].index)

                # Swap the detections
                df.loc[idx] = video_df.loc[swap_idx]

                # Mark both detections as swapped
                df.loc[idx, 'da_swapped'] = True
                df.loc[idx, 'track_id'] = original_track_id
        return df
