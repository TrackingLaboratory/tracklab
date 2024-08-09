"""
On veut un nn.Module en pytorch

Le detection tokenizer :
x est l'input à la forward
x.feats est un dict composé des keys "embeddings", "visibility_scores", "bbox_ltwh", "bbox_score", "keypoints_xyc"
On veut un MLP (ou une projection linéaire) qui envoie ces features vers une dimension token_dim. On ne veut pas de couche de normalisation.

L'idée est d'appliquer ceci pour l'ensemble des detections qui composent x.
Une fois ces tokens obtenus, on garde les tokens liés aux detections. Ils deviendront x.tokens des detections.
Pour ce qui est des tokens liés aux tracks, on les utilisera pour les encoder via un encoder.

L'encoder de la tracklets  :
On aligne chacune des detections liées à une tracklet auquels on vient encoder les timestamps respectifs de ses observations.
(sin + cos positional encoding ?)
On passe ces detections + un token class_0 dans un encoder (transformer) qui va encoder ces données.
On récupère en output le token class_0 qui représente un descripteur de la tracklet.
C'est lui qui deviendra le x.embs associé.
"""

import torch
import torch.nn as nn
import math

from .transformers import Module
from ..simformer import Detections, Tracklets

class LinearProjection(Module):
    def __init__(self, token_dim: int, feat_dim: int = 3133, tokenizer_checkpoint_path: str = None):
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        # Define a linear layer to project features to token_dim
        self.linear = nn.Linear(feat_dim, token_dim)

    def forward(self, x):
        # Concatenate all the feature dimensions
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
        # Project the concatenated features to token_dim
        tokens[x.feats_masks] = self.linear(cat_feats[x.feats_masks])
        return tokens
    
class MLP(Module):
    def __init__(self, token_dim: int, feat_dim: int = 3133, tokenizer_checkpoint_path: str = None):
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        
        # Define a Multi-Layer Perceptron to project features to token_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, token_dim)
        )

    def forward(self, x):
        # Concatenate all the feature dimensions
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
        # Project the concatenated features to token_dim using the MLP
        tokens[x.feats_masks] = self.mlp(cat_feats[x.feats_masks])
        return tokens


class Tokenizer(Module):
    def __init__(self, token_dim, feat_dim, emb_dim, n_heads, n_layers, num_registers: int = 3, dim_feedforward=2048, dropout=0.1, checkpoint_path: str = None, tokenizer_checkpoint_path: str = None,
        **kwargs):
        super().__init__()
        self.lin_proj = MLP(token_dim, feat_dim, tokenizer_checkpoint_path)
        self.token_dim = token_dim
        self.feat_dim = feat_dim

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.pos_encoder = PositionalEncoding(emb_dim)
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.special_tokens = nn.Parameter(torch.randn(num_registers + 1, 1, emb_dim))
        self.num_registers = num_registers

        self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.tokenizer")

    def forward(self, x):
        tokens = self.lin_proj(x)
        if isinstance(x, Detections):
            return tokens.squeeze(dim=2)
        else:
            assert isinstance(x, Tracklets), "Input must be either Detections or Tracklets."

        return tokens.mean(dim=2)
        # Apply positional encoding
        #src = self.pos_encoder(tokens, x.feats["age"], x.feats_masks)
        #
        ## Reshape input to (S, B*N, E)
        #B, N, S, E = tokens.shape
        #src = src.permute(2, 0, 1, 3).reshape(S, B * N, E)
        #
        ## Add class_0 token
        #special_tokens = self.special_tokens.expand(self.num_registers+1, B*N, E)
        #src = torch.cat([special_tokens, src], dim=0)
        #
        ## Update mask to include class_0 token
        #new_mask = torch.ones((B * N, self.num_registers + 1), dtype=torch.bool, device=x.feats_masks.device)
        #src_mask = torch.cat([new_mask, x.feats_masks.reshape(B * N, -1)], dim=1)
        #
        ## Apply transformer encoder
        #output = self.transformer_encoder(src, src_key_padding_mask=~src_mask)
        #
        ## Extract class_0 token
        #class_0_output = output[0].reshape(B, N, E)
        #return class_0_output


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=200):
        super().__init__()
        self.emb_dim = emb_dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, age, mask):
        B, N, S, E = x.shape
        x = x.view(B * N, S, E)
        age = age.view(B * N, S).to(torch.long)
        mask = mask.view(B * N, S)

        # Ajouter l'encodage positionnel à x
        x[mask] = x[mask] + self.pe[age[mask]]

        return x.view(B, N, S, E)