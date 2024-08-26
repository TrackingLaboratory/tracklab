import torch

from dd_sort.simformer.transforms import BatchTransform


class FeatsDetDropout(BatchTransform):
    """
    Apply some dropout (replaced by zeros) on the features listed in `allowed_feats` with probability `p_drop`.
    If `on_track_only` is True, the dropout is applied only on the track features (not on the detection features).

    batch : dict of dicts
        det_feats : dict of torch.Tensor [B, N, ...] (padded are nan)
        det_targets : torch.Tensor [B, N] (padded are nan)
        track_feats : dict of torch.Tensor [B, N, T, ...] (padded are nan)
        track_targets : torch.Tensor [B, N, T] (padded are nan)
    """

    def __init__(self, p_drop: float = 0.1, allowed_feats: list = [], on_track_only: bool = False):
        super().__init__()
        self.p_drop = p_drop
        self.allowed_feats = allowed_feats
        self.on_track_only = on_track_only

    def __call__(self, batch):
        if 'det_feats' in batch and not self.on_track_only:
            batch['det_feats'] = self._apply_dropout(batch['det_feats'], batch['det_targets'])

        if 'track_feats' in batch:
            batch['track_feats'] = self._apply_dropout(batch['track_feats'], batch['track_targets'])
        return batch

    def _apply_dropout(self, features, targets):
        """
        Apply dropout to the given features with probability self.p_drop
        """
        drop_mask = (torch.rand(targets.shape, device=targets.device) < self.p_drop) & (~targets.isnan())
        dropped_feat = torch.randint(0, len(list(set(self.allowed_feats) & set(features.keys()))), targets.shape,
                                            device=targets.device)
        for i, feat in enumerate(features):
            if feat in self.allowed_feats:
                features[feat][drop_mask & (dropped_feat == i)] = 0.
        return features
