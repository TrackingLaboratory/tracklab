from typing import List

import torch
import torch.nn as nn
import torchvision

from .transformers import Module


class LinearClassifier(Module):
    def __init__(self, emb_dim: int, checkpoint_path: str = None, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim

        self.linear = nn.Linear(emb_dim, 1)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="classifier")

    def forward(self, dets):
        confs = torch.zeros((dets.embs.shape[0], dets.embs.shape[1], 1), device=dets.embs.device)
        confs[dets.masks] = self.linear(dets.embs[dets.masks])
        dets.confs = confs.squeeze(dim=2)
        return dets


class MLPClassifier(Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: List[int],
        activation_fn: str = "relu",
        dropout: int = 0.1,
        checkpoint_path: str = None,
        **kwargs
    ):
        assert hidden_dim[-1] == 1, "Last hidden dim must be 1"
        super().__init__()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        if activation_fn == "relu":
            self.activation_fn = torch.nn.modules.activation.ReLU
        elif activation_fn == "gelu":
            self.activation_fn = torch.nn.modules.activation.GELU
        else:
            raise NotImplementedError
        self.dropout = dropout

        self.mlp = torchvision.ops.MLP(
            in_channels=emb_dim,
            hidden_channels=hidden_dim,
            norm_layer=None,
            activation_layer=self.activation_fn,
            dropout=dropout,
            inplace=False,
        )

        self.init_weights(checkpoint_path=checkpoint_path, module_name="classifier")

    def forward(self, dets):
        confs = torch.zeros((dets.embs.shape[0], dets.embs.shape[1], 1), device=dets.embs.device)
        confs[dets.masks] = self.mlp(dets.embs[dets.masks])
        dets.confs = confs.squeeze(dim=2)
        return dets
