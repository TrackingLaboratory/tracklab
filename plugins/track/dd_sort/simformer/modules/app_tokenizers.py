import torch
import torch.nn as nn

from .transformers import Module


class LinearAppearance(Module):
    default_similarity_metric = "cosine"

    def __init__(
        self,
        token_dim: int,
        feat_dim: int = 1799,
        alpha: int = 0.9,
        enable_ll: bool = True,
        agg_strat: str = "ema",
        checkpoint_path: str = None,
        **kwargs
    ):
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        self.enable_ll = enable_ll
        self.alpha = alpha
        if agg_strat == "ema":
            self.aggregation_fn = self.ema
        elif agg_strat == "mean":
            self.aggregation_fn = self.mean
        elif agg_strat == "last":
            self.aggregation_fn = self.last
        else:
            raise NotImplementedError

        if self.enable_ll:
            self.linear = nn.Linear(feat_dim, token_dim)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.LinearAppearance")

    def forward(self, x):
        feats, masks = x.feats, x.feats_masks
        # feats = torch.cat([feats["embeddings"], feats["visibility_scores"]], dim=3)
        feats = feats["embeddings"]
        if self.enable_ll:
            tokens = torch.zeros(
                (feats.shape[0], feats.shape[1], feats.shape[2], self.token_dim),
                device=feats.device,
                dtype=torch.float32,
            )
            tokens[masks] = self.linear(feats[masks])
        else:
            tokens = feats
        if masks.shape[2] > 1:
            tokens = self.aggregation_fn(tokens, masks, alpha=self.alpha)
        tokens = tokens.squeeze(dim=2)
        return tokens

    # def mean(self, tokens, masks):  # created an error
    #     weights = masks.float()
    #     weights = (weights / weights.sum(dim=2, keepdim=True)).nan_to_num(0.0)
    #     mean = torch.sum(tokens * weights.unsqueeze(dim=3), dim=2, keepdim=True)
    #     return mean

    def mean(self, tokens, masks, **kwargs):
        # Ensure masks are of type float and unsqueezed for broadcasting over the token dimension D
        masks = masks.unsqueeze(-1).float()  # Shape [B, N, T, 1]
        # Replace NaNs in tokens with zeros to avoid propagation of NaNs
        tokens = torch.nan_to_num(tokens, nan=0.0)
        # Calculate the sum of valid tokens (by multiplying with the mask)
        masked_tokens_sum = tokens * masks  # Shape [B, N, T, D]
        # Compute the valid count along the T dimension (we sum the mask)
        valid_count = masks.sum(dim=2)  # Shape [B, N, 1], this is the count of valid tokens
        # Avoid division by zero by replacing 0 counts with 1 in valid_count
        valid_count = valid_count.clamp(min=1.0)  # Shape [B, N, 1]
        # Calculate the mean of valid tokens by summing along T and dividing by the valid count
        tokens_mean = masked_tokens_sum.sum(dim=2) / valid_count  # Shape [B, N, D]
        return tokens_mean

    # def ema(self, tokens, masks, alpha=0.9):  # old version, was not working? slow too
    #     # fixme batch it
    #     ema = torch.zeros_like(tokens, device=tokens.device)
    #     for b in range(tokens.shape[0]):
    #         for n in range(tokens.shape[1]):
    #             for t in reversed(range(tokens.shape[2])):
    #             # for t in range(tokens.shape[2]):
    #                 if masks[b, n, t]:
    #                     ema[b, n, 0] = alpha * tokens[b, n, t] + (1 - alpha) * ema[b, n, 0]
    #     ema = ema.sum(dim=2, keepdim=True)
    #     return ema

    def ema(self, tokens, masks, alpha=0.9):  # TODO: make it work with multiple part embes and vis scores
        # Ensure masks are of type float and unsqueezed for broadcasting over the token dimension D
        masks = masks.unsqueeze(-1).float()  # Shape [B, N, T, 1]

        # Replace NaNs in tokens with zeros to avoid propagation of NaNs
        tokens = torch.nan_to_num(tokens, nan=0.0)

        # Initialize EMA tensor to hold the result
        B, N, T, D = tokens.shape

        # Initialize the EMA starting from the last valid token
        ema_t = torch.zeros((B, N, D), dtype=tokens.dtype, device=tokens.device)

        # A boolean tensor to track if ema_t has been initialized with the first valid token
        initialized = torch.zeros((B, N, 1), dtype=torch.bool, device=tokens.device)

        # Traverse the time dimension in reverse (from T-1 to 0)
        for t in reversed(range(T)):
            # Current token and mask at time step t
            token_t = tokens[:, :, t, :]  # Shape [B, N, D]
            mask_t = masks[:, :, t, :]  # Shape [B, N, 1]

            # First valid token should fully replace ema_t if alpha is 0
            if alpha == 0:
                # Update ema_t with the first valid token
                ema_t = torch.where(~initialized & (mask_t > 0), token_t, ema_t)

                # Mark that ema_t has been initialized
                initialized = initialized | (mask_t > 0)
            else:
                # For alpha > 0, perform standard EMA update
                ema_t = mask_t * (alpha * token_t + (1 - alpha) * ema_t) + (1 - mask_t) * ema_t

        return ema_t


    def last(self, tokens, masks, **kwargs):
        last = tokens[:, :, 0]  # detections in reverse order: first one is the most recent one
        assert not torch.isnan(last).any()
        assert masks[:, :, 0].all()
        return last


class SmartLinearAppearance(Module):
    def __init__(self, token_dim: int, feat_dim: int = 1799, enable_ll: bool = True, checkpoint_path: str = None, alpha=0.9, **kwargs):
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        self.enable_ll = enable_ll
        self.linear = nn.Linear(feat_dim, token_dim)
        self.alpha = alpha

        self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.SmartLinearAppearance")

    def forward(self, x):
        embs, vis, masks = x.feats["embeddings"], x.feats["visibility_scores"], x.feats_masks
        if masks.shape[2] > 1:
            embs, vis, masks = self.smart(embs, vis, masks, self.alpha)
        # feats = torch.cat([embs, vis], dim=3)  # keep modalities separate: vis score cannot be simply appended
        feats = embs
        if self.enable_ll:
            tokens = torch.zeros(
                (embs.shape[0], embs.shape[1], 1, self.token_dim),
                device=feats.device,
                dtype=torch.float32,
            )
            tokens[masks] = self.linear(feats[masks])
        else:
            tokens = feats
        tokens = tokens.squeeze(dim=2)
        return tokens

    def smart(self, embs, vis, masks, alpha=0.9):
        """
        :param embs: [B, N, T, 1792]
        :param vis: [B, N, T, 7]
        :param masks: [B, N, T]
        :return: new_embs [B, N, 1, 1792], new_vis [B, N, 1, 7], new_masks [B, N, 1]
        """
        if embs.shape[-1] == 1792:  # FIXME temporary fix to handle two different reid models
            feature_dim = 256
            num_parts = 7
        elif embs.shape[-1] == 3072:
            feature_dim = 512
            num_parts = 6
        elif embs.shape[-1] == 512:
            feature_dim = 256
            num_parts = 2
        elif embs.shape[-1] == 512:
            feature_dim = 256
            num_parts = 2
        else:
            feature_dim = 128
            num_parts = 1
        new_embeddings = torch.zeros(
            (embs.shape[0], embs.shape[1], 1, embs.shape[3]),
            device=embs.device,
            dtype=torch.float32,
        )
        new_vis = torch.zeros(
            (embs.shape[0], embs.shape[1], 1, vis.shape[3]), device=embs.device, dtype=torch.float32
        )
        new_masks = torch.zeros(
            (embs.shape[0], embs.shape[1], 1), device=embs.device, dtype=torch.bool
        )
        for b in range(masks.shape[0]):
            for n in range(masks.shape[1]):
                for t in reversed(range(masks.shape[2])):
                    if masks[b, n, t]:
                        # For a given body part P, if:
                        # - P is visible in both the tracklet and the detection, xor = False and ema_scores_tracklet=ema_alpha and ema_scores_detection=1-ema_alpha -> both tracklet and det features are used in a normal EMA update step
                        # - P is visible in the tracklet but not in the detection, xor = True and ema_scores_tracklet=1 and ema_scores_detection=0 -> smooth_feat=tracklet_features
                        # - P is visible in the detection but not in the tracklet, xor = True and ema_scores_tracklet=0 and ema_scores_detection=1 -> smooth_feat=detection_features
                        # - P is not visible in both the tracklet and the detection, xor = False and ema_scores_tracklet=0 and ema_scores_detection=0 -> smooth_feat=1 (TODO why?)

                        tracklet_features = new_embeddings[b, n, 0].reshape((num_parts, feature_dim))
                        detection_features = embs[b, n, t].reshape((num_parts, feature_dim))

                        xor = torch.logical_xor(new_vis[b, n, 0], vis[b, n, t])
                        ema_scores_tracklet = (
                            new_vis[b, n, 0] * vis[b, n, t]
                        ) * alpha + xor * new_vis[b, n, 0]
                        ema_scores_detection = (new_vis[b, n, 0] * vis[b, n, t]) * (
                            1 - alpha
                        ) + xor * vis[b, n, t]
                        smooth_feat = (
                            ema_scores_tracklet.unsqueeze(dim=1) * tracklet_features
                            + ema_scores_detection.unsqueeze(dim=1) * detection_features
                        )
                        new_embeddings[b, n, 0] = smooth_feat.reshape(-1)

                        smooth_visibility_scores = torch.maximum(new_vis[b, n, 0], vis[b, n, t])
                        new_vis[b, n, 0] = smooth_visibility_scores

                        new_masks[b, n, 0] = True
        return new_embeddings, new_vis, new_masks
