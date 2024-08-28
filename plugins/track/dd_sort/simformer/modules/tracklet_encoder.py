import torch
import torch.nn as nn
import math

from .transformers import Module
from ..simformer import Detections


class LinearSum(Module):
    def __init__(self, app_feat_dim: int, st_feat_dim: int, token_dim: int, **kwargs):
        super().__init__()
        self.app_feat_dim = app_feat_dim
        self.st_feat_dim = st_feat_dim
        self.token_dim = token_dim
        self.app_linear = nn.Linear(app_feat_dim, token_dim)
        self.st_linear = nn.Linear(st_feat_dim, token_dim)

    def forward(self, x):
        app_cat_feats = torch.cat(
            [x.feats["embeddings"],
             x.feats["visibility_scores"]],
            dim=-1
        )
        app_tokens = torch.zeros(
            (*app_cat_feats.shape[:-1], self.token_dim),
            device=app_cat_feats.device,
            dtype=torch.float32,
        )
        st_cat_feats = torch.cat(
            [x.feats["bbox_ltwh"],
             x.feats["keypoints_xyc"].reshape(*x.feats_masks.shape, -1)],
            dim=-1
        )
        st_tokens = torch.zeros(
            (*st_cat_feats.shape[:-1], self.token_dim),
            device=st_cat_feats.device,
            dtype=torch.float32,
        )
        app_tokens[x.feats_masks] = self.app_linear(app_cat_feats[x.feats_masks])
        st_tokens[x.feats_masks] = self.st_linear(st_cat_feats[x.feats_masks])
        tokens = app_tokens + st_tokens
        return tokens


class LinearProjection(Module):
    """
    Project features of detections from feat_dim to token_dim using a linear projection.
    """

    def __init__(self, feat_dim: int, token_dim: int, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self.token_dim = token_dim
        self.linear = nn.Linear(feat_dim, token_dim)

    def forward(self, x):
        cat_feats = torch.cat(
            [x.feats["embeddings"],
             x.feats["visibility_scores"],
             x.feats["bbox_ltwh"],
             x.feats["keypoints_xyc"].reshape(*x.feats_masks.shape, -1)],
            dim=-1
        )
        tokens = torch.zeros(
            (*cat_feats.shape[:-1], self.token_dim),
            device=cat_feats.device,
            dtype=torch.float32,
        )
        tokens[x.feats_masks] = self.linear(cat_feats[x.feats_masks])
        return tokens


class MLP(Module):
    """
    Project features of detections from feat_dim to token_dim using an MLP.
    """

    def __init__(self, feat_dim: int, token_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, token_dim)
        )

    def forward(self, x):
        cat_feats = torch.cat(
            [x.feats["embeddings"],
             x.feats["visibility_scores"],
             x.feats["bbox_ltwh"],
             x.feats["keypoints_xyc"].reshape(*x.feats_masks.shape, -1)],
            dim=-1
        )
        tokens = torch.zeros(
            (*cat_feats.shape[:-1], self.token_dim),
            device=cat_feats.device,
            dtype=torch.float32,
        )
        tokens[x.feats_masks] = self.mlp(cat_feats[x.feats_masks])
        return tokens


class TrackletEncoder(Module):
    """
    Tokenize detections or tracklets into a single token.
    Detections are simply projected into a token space using det_tokenizer.
    Tracklets are projected into a token space using the projected detections by det_tokenizer and then encoded using a
    transformer encoder.
    """

    def __init__(self, emb_dim, det_tokenizer, n_heads: int = 8, n_layers: int = 1, dim_feedforward=4096,
                 num_registers: int = 3,
                 dropout=0.1, checkpoint_path: str = None, use_only_det_tokenizer: bool = False, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.num_registers = num_registers
        self.dropout = dropout
        self.use_only_det_tokenizer = use_only_det_tokenizer

        self.det_tokenizer = det_tokenizer
        if not use_only_det_tokenizer:
            self.pos_encoder = PositionalEncoding(emb_dim)
            encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward, dropout)
            self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
            self.registers_tokens = nn.Parameter(torch.randn(num_registers + 1, 1, emb_dim))

        self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.TrackletEncoder")

    def forward(self, x):
        # handle the detections by simply projection them into a token space
        tokens = self.det_tokenizer(x)
        if isinstance(x, Detections):
            # job is done for the detections
            return tokens.squeeze(dim=2)

        # handle the tracklets
        if self.use_only_det_tokenizer:
            # return the mean of the detections (this is used to train the det_tokenizer)
            return masked_mean(tokens, x.feats_masks, dim=2)

        B, N, S, E = tokens.shape

        src = self.pos_encoder(tokens, x.feats["age"], x.feats_masks)
        src = src.permute(2, 0, 1, 3).reshape(S, B * N, E)

        registers_tokens = self.registers_tokens.expand(self.num_registers + 1, B * N, E)
        src = torch.cat([src, registers_tokens], dim=0)

        new_mask = torch.ones((B * N, self.num_registers + 1), dtype=torch.bool, device=x.feats_masks.device)
        src_mask = torch.cat([x.feats_masks.reshape(B * N, S), new_mask], dim=1)

        output = self.encoder(src, src_key_padding_mask=~src_mask)  # [S, B*N, E]
        output = output.permute(1, 0, 2).reshape(B, N, S + self.num_registers + 1, E)[:, :, -1, :]  # [B, N, E]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=1000):
        super().__init__()
        self.emb_dim = emb_dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x, age, mask):
        B, N, S, E = x.shape
        x = x.view(B * N, S, E)
        age = age.view(B * N, S).to(torch.long)
        mask = mask.view(B * N, S)
        age[mask] = age[mask].clamp(min=0, max=self.max_len)

        x[mask] = x[mask] + self.pe[age[mask]]

        return x.view(B, N, S, E)


def masked_mean(tensor, mask, dim):
    # Replace masked values with 0
    masked_tensor = torch.where(mask.unsqueeze(-1), tensor, torch.zeros_like(tensor))

    # Sum along the specified dimension
    sum_result = torch.sum(masked_tensor, dim=dim)

    # Count non-masked elements
    count = torch.sum(mask, dim=dim).unsqueeze(-1)

    # Avoid division by zero
    count = torch.clamp(count, min=1)

    # Calculate mean
    mean_result = sum_result / count

    return mean_result


"""
Rest of the file is kept for journaling the experiments and the code that was tried and did not work.

# return masked_mean(tokens, x.feats_masks, dim=2)  # torch.sum(tokens, dim=2)/torch.sum(x.feats_masks.unsqueeze(-1), dim=2)  # tokens.mean(dim=2)
# The problem is that even with that we do not have good perfs : IDF1 44.559 & AssA 25.95 


# next steps :
# train using MLP projection (done)
# return last detection from tracklet (done)
# return tracklet.mean(done)
# use encoder w/ 1 special token(done) 62
# add pose encoding (done) 65
# try on class_0 (doing) 67
# try to train from scratch like that (doing) 68 - does not work
# add layers to transformer 70 : tried with 2 (ok-ish) - 71 : 4 (need way more training)
# (add more special tokens)

# Apply positional encoding
# src = self.pos_encoder(tokens, x.feats["age"], x.feats_masks)
## Reshape input to (S, B*N, E)
# B, N, S, E = tokens.shape
# src = src.permute(2, 0, 1, 3).reshape(S, B * N, E)
#
## Add class_0 token
# registers_tokens = self.registers_tokens.expand(self.num_registers+1, B*N, E)
# src = torch.cat([registers_tokens, src], dim=0)
#
## Update mask to include class_0 token
# new_mask = torch.ones((B * N, self.num_registers + 1), dtype=torch.bool, device=x.feats_masks.device)
# src_mask = torch.cat([new_mask, x.feats_masks.reshape(B * N, -1)], dim=1)
#
## Apply transformer encoder
# output = self.encoder(src, src_key_padding_mask=~src_mask)
#
## Extract class_0 token
# class_0_output = output[0].reshape(B, N, E)
# output = output.permute(1, 0, 2).reshape(B, N, S+1, E)[:, :, :-1, :]
# mean1 = masked_mean(output, x.feats_masks, dim=2)
# return class_0_output
"""
