import torch
import torch.nn as nn

from .transformers import Module


class LinearAppearance(Module):
    def __init__(
        self,
        token_dim: int,
        feat_dim: int = 1799,
        enable_ll: bool = True,
        agg_strat: str = "ema",
        checkpoint_path: str = None,
        **kwargs
    ):
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        self.enable_ll = enable_ll
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
        feats = torch.cat([feats["embeddings"].flatten(start_dim=-2, end_dim=-1), feats["visibility_scores"]], dim=3)
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
            tokens = self.aggregation_fn(tokens, masks)
        tokens = tokens.squeeze(dim=2)
        return tokens

    def mean(self, tokens, masks):
        weights = masks.float()
        weights = (weights / weights.sum(dim=2, keepdim=True)).nan_to_num(0.0)
        mean = torch.sum(tokens * weights.unsqueeze(dim=3), dim=2, keepdim=True)
        return mean

    def ema(self, tokens, masks, alpha=0.9):
        # fixme batch it
        ema = torch.zeros_like(tokens, device=tokens.device)
        for b in range(tokens.shape[0]):
            for n in range(tokens.shape[1]):
                for t in range(tokens.shape[2]):
                    if masks[b, n, t]:
                        ema[b, n, 0] = alpha * tokens[b, n, t] + (1 - alpha) * ema[b, n, 0]
        ema = ema.sum(dim=2, keepdim=True)
        return ema

    def last(self, tokens, masks):
        # fixme batch it
        last = torch.zeros_like(tokens, device=tokens.device)
        for b in range(tokens.shape[0]):
            for n in range(tokens.shape[1]):
                for t in range(tokens.shape[2]):
                    if masks[b, n, t]:
                        last[b, n, 0] = tokens[b, n, t]
        last = last.sum(dim=2, keepdim=True)
        return last


class SmartLinearAppearance(Module):
    def __init__(self, token_dim: int, feat_dim: int = 1799, enable_ll: bool = True, checkpoint_path: str = None, **kwargs):
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        self.enable_ll = enable_ll
        self.linear = nn.Linear(feat_dim, token_dim)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.SmartLinearAppearance")

    def forward(self, x):
        embs, vis, masks = x.feats["embeddings"], x.feats["visibility_scores"], x.feats_masks
        if masks.shape[2] > 1:
            embs, vis, masks = self.smart(embs, vis, masks)
        feats = torch.cat([embs.flatten(start_dim=-2, end_dim=-1), vis], dim=3)
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
        :param embs: [B, N, T, num_parts, emb_dim]
        :param vis: [B, N, T, num_parts]
        :param masks: [B, N, T]
        :return: new_embs [B, N, 1, num_parts, emb_dim], new_vis [B, N, 1, num_parts], new_masks [B, N, 1]
        """
        # FIXME batch it
        new_embeddings = torch.zeros(
            (embs.shape[0], embs.shape[1], 1, embs.shape[-2], embs.shape[-1]),
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
                for t in range(masks.shape[2]):
                    if masks[b, n, t]:
                        tracklet_features = new_embeddings[b, n, 0]
                        detection_features = embs[b, n, t]

                        xor = torch.logical_xor(new_vis[b, n, 0], vis[b, n, t])
                        ema_scores_tracklet = (
                            new_vis[b, n, 0] * vis[b, n, t]
                        ) * alpha + xor * new_vis[b, n, 0]
                        ema_scores_detection = (new_vis[b, n, 0] * vis[b, n, t]) * (
                            1 - alpha
                        ) + xor * vis[b, n, t]
                        smooth_feat = (
                            ema_scores_tracklet.unsqueeze(1) * tracklet_features
                            + ema_scores_detection.unsqueeze(1) * detection_features
                        )
                        new_embeddings[b, n, 0] = smooth_feat

                        smooth_visibility_scores = torch.maximum(new_vis[b, n, 0], vis[b, n, t])
                        new_vis[b, n, 0] = smooth_visibility_scores

                        new_masks[b, n, 0] = True
        return new_embeddings, new_vis, new_masks
