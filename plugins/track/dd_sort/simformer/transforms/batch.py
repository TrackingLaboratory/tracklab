import logging

import torch

from dd_sort.simformer.transforms import BatchTransform

log = logging.getLogger(__name__)


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

    def __init__(self, p_drop: float = 0.1, allowed_feats: list = [],
                 on_track_only: bool = False, dropout_type: str = "random"):
        super().__init__()
        self.p_drop = p_drop
        self.allowed_feats = allowed_feats
        self.on_track_only = on_track_only
        self.dropout_type = dropout_type

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
                min_feat, max_feat = nanmin(features[feat]), nanmax(features[feat])
                if self.dropout_type == "random" and min_feat < max_feat:
                    values = torch.distributions.uniform.Uniform(min_feat, max_feat).sample(features[feat][drop_mask & (dropped_feat == i)].shape)
                else:
                    values = 0.
                features[feat][drop_mask & (dropped_feat == i)] = values
        return features

def nanmax(tensor):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max()
    return output


def nanmin(tensor):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min()
    return output


class AppEmbNoise(BatchTransform):
    """
    Apply Gaussian noise to the track feature embeddings in the batch. The amount of noise added
    is controlled by the `alpha` parameter, which scales the noise by the standard deviation of
    the embeddings along the last dimension.

    The noise added to the embeddings is computed as:
    - `noise = alpha * std * N(0, 1)` where `std` is the standard deviation of the embeddings
      along the last dimension, and `N(0, 1)` is standard Gaussian noise.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def __call__(self, batch):
        embs = batch["track_feats"]["embeddings"]  # [B, N, T, D]
        std = embs.std(dim=-1, keepdim=True)
        noise = torch.randn_like(embs, device=embs.device)
        batch["track_feats"]["embeddings"] = embs + self.alpha * std * noise
        return batch


class BBoxShake(BatchTransform):
    """
    Apply a data augmentation technique that slightly perturbs the bounding box coordinates.
    This "shake" effect is controlled by an `alpha` parameter, which influences how much
    the bounding boxes are modified.

    The modifications are as follows:
    - `top` coordinate is adjusted by a random value from the uniform distribution
      between `-alpha * height` and `+alpha * height`.
    - `left` coordinate is adjusted by a random value from the uniform distribution
      between `-alpha * width` and `+alpha * width`.
    - `width` is scaled by a factor of `1 + Uniform(-alpha, +alpha)`.
    - `height` is scaled by a factor of `1 + Uniform(-alpha, +alpha)`.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def __call__(self, batch):
        bboxes = batch["track_feats"]["bbox_ltwh"]  # [B, N, T, 4]

        # Split the bboxes into components
        left, top, width, height = bboxes.unbind(dim=-1)

        # Generate uniform noise for the modifications
        delta_left = torch.empty_like(left, device=bboxes.device).uniform_(-self.alpha, self.alpha)
        delta_top = torch.empty_like(top, device=bboxes.device).uniform_(-self.alpha, self.alpha)
        delta_width = torch.empty_like(width, device=bboxes.device).uniform_(-self.alpha, self.alpha)
        delta_height = torch.empty_like(height, device=bboxes.device).uniform_(-self.alpha, self.alpha)

        # Apply the noise to the bounding box components
        shaken_left = left + (delta_left * width)
        shaken_top = top + (delta_top * height)
        shaken_width = width * (1 + delta_width)
        shaken_height = height * (1 + delta_height)

        # Reassemble the shaken bounding boxes
        shaken_bboxes = torch.stack([shaken_left, shaken_top, shaken_width, shaken_height], dim=-1)

        # Clamp the values to stay within the [0, 1] range
        shaken_bboxes = torch.clamp(shaken_bboxes, 0.0, 1.0)

        batch["track_feats"]["bbox_ltwh"] = shaken_bboxes
        return batch


class KeypointsShake(BatchTransform):
    """
    Apply a data augmentation technique that slightly perturbs the keypoint coordinates (x, y)
    based on the height and width of the corresponding bounding box, while keeping them within
    the image boundaries. The `confidence` values are not modified.

    The modifications are as follows:
    - `x` coordinate is adjusted by a random value from the uniform distribution
      between `-alpha * width` and `+alpha * width`.
    - `y` coordinate is adjusted by a random value from the uniform distribution
      between `-alpha * height` and `+alpha * height`.
    """

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def __call__(self, batch):
        keypoints = batch["track_feats"]["keypoints_xyc"]  # [B, N, T, 17, 3]
        bboxes = batch["track_feats"]["bbox_ltwh"]  # [B, N, T, 4]

        # Split the keypoints into x, y, and confidence components
        x, y, c = keypoints[..., 0], keypoints[..., 1], keypoints[..., 2]

        # Split the bounding boxes into left, top, width, height
        _, _, width, height = bboxes.unbind(dim=-1)

        # Generate uniform noise for the x and y coordinates based on bbox width and height
        delta_x = torch.empty_like(x, device=keypoints.device).uniform_(-self.alpha, self.alpha)
        delta_y = torch.empty_like(y, device=keypoints.device).uniform_(-self.alpha, self.alpha)

        # Apply the noise to the x and y coordinates
        shaken_x = x + delta_x * width[..., None]
        shaken_y = y + delta_y * height[..., None]

        # Clamp the x and y coordinates to be within the image bounds [0, 1]
        shaken_x = torch.clamp(shaken_x, 0.0, 1.0)
        shaken_y = torch.clamp(shaken_y, 0.0, 1.0)

        # Reassemble the keypoints with the shaken x, y, and original confidence c
        shaken_keypoints = torch.stack([shaken_x, shaken_y, c], dim=-1)

        batch["track_feats"]["keypoints_xyc"] = shaken_keypoints
        return batch


class SwapDetections(BatchTransform):

    def __init__(self, swap_prob=0.1, max_proportion=0.3):
        super().__init__()
        self.swap_prob = swap_prob
        self.max_prop = max_proportion

    def __call__(self, batch):
        batch_ages = batch["track_feats"]["age"]
        swap_mask = torch.rand(batch_ages.shape, device=batch_ages.device) < self.swap_prob
        prop = self.max_prop
        indices_over = ((swap_mask.count_nonzero(dim=2)/swap_mask.shape[2])[:,:,0]>prop).nonzero()
        for idxs in indices_over:
            a = swap_mask[idxs[0], idxs[1], :, 0]
            indices = torch.nonzero(a)[:,0]
            excess = int(len(indices) - swap_mask.shape[2]*prop)
            indices = indices[:excess]
            swap_mask[idxs[0], idxs[1], indices, 0] = False
        for b_idx in range(batch_ages.shape[0]):
            unique_ages = torch.unique(batch_ages[b_idx])
            unique_ages = unique_ages[~torch.isnan(unique_ages)]
            for age in unique_ages:
                age_mask = ((batch_ages[b_idx] == age) & swap_mask[b_idx])[:, :, 0]
                positions = age_mask.nonzero()
                # log.info(f"pos ({len(positions)}): {positions}")
                if len(positions) < 2:
                    log.debug(f"Not enough positions to swap age {age} sample {b_idx}")
                    continue
                perm = torch.randperm(len(positions))
                while not torch.all(perm[:-1]>perm[1:]):
                    perm = torch.randperm(len(positions))
                shuffled_positions = positions[perm]
                track_ids = batch["track_targets"]

                for k, v in batch["track_feats"].items():
                    if v.dim() == 3:
                        v[b_idx, positions[:, 0], positions[:, 1]] = v[
                                                                        b_idx,
                                                                        shuffled_positions[
                                                                        :, 0],
                                                                        shuffled_positions[
                                                                        :, 1]]
                    else:
                        v[b_idx, positions[:, 0], positions[:, 1], :] = v[
                            b_idx, shuffled_positions[:, 0], shuffled_positions[:, 1], :]

        return batch