import logging
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import MLP
log = logging.getLogger(__name__)


class Module(nn.Module):
    def __init__(self, checkpoint_path: str = None, **kwargs):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def init_weights(self, checkpoint_path: str = None, module_name: str = None):
        if checkpoint_path and module_name:
            state_dict = torch.load(checkpoint_path)["state_dict"]
            state_dict = OrderedDict(
                (k, state_dict[k]) for k in state_dict if k.startswith(module_name)
            )
            state_dict = OrderedDict(
                (k.replace(module_name + ".", ""), v) for k, v in state_dict.items()
            )
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            log.info(f"Loaded checkpoint weights for {module_name} from `{checkpoint_path}`.")
            if missing:
                log.warning(f"Missing keys while loading: {missing}. Initializing random weights for those.")
            if unexpected:
                log.warning(f"Unexpected keys while loading: {unexpected}. Initializing random weights for those.")
            params_to_init = missing + unexpected
        else:
            params_to_init = self.named_modules()
        modules = dict(self.named_modules())
        for key in params_to_init:
            if key in modules:
                layer = modules[key]
                if layer.dim() > 1:
                    nn.init.xavier_uniform_(layer)
                else:
                    nn.init.uniform_(layer)


class Identity(Module):
    def __init__(self, checkpoint_path: str = None, **kwargs):
        super().__init__()
        self.init_weights(checkpoint_path=checkpoint_path, module_name="transformer")

    def forward(self, tracks, dets):
        dets.embs = dets.tokens
        tracks.embs = tracks.tokens
        return tracks, dets


class Perceptron(Module):
    def __init__(self,
                 emb_dim: int = 1024,
                 n_layers: int = 6,
                 dim_feedforward: int = 4096,
                 checkpoint_path: str = None,
                 **kwargs,
                 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.mlp = MLP(
            in_channels=self.emb_dim,
            hidden_channels=[self.dim_feedforward]*(self.n_layers-1)+[self.emb_dim],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=True,
        )
        self.init_weights(checkpoint_path=checkpoint_path, module_name="transformer")

    def forward(self, tracks, dets):
        dets_embs = self.mlp(dets.tokens.flatten(0, 1))
        tracks_embs = self.mlp(tracks.tokens.flatten(0, 1))
        dets.embs = dets_embs.reshape(dets.tokens.shape[0], dets.tokens.shape[1], -1)
        tracks.embs = tracks_embs.reshape(tracks.tokens.shape[0], tracks.tokens.shape[1], -1)
        return tracks, dets


class Encoder(Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 4096,
        dropout: int = 0.1,
        activation_fn: str = "gelu",
        checkpoint_path: str = None,
        use_processed_track_tokens: bool = True,
        src_key_padding_mask: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.use_processed_track_tokens = use_processed_track_tokens
        self.src_key_padding_mask = src_key_padding_mask

        self.det_encoder = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        self.track_encoder = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        self.cls = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        self.src_norm = nn.LayerNorm(emb_dim)
        self.src_drop = nn.Dropout(dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            self.emb_dim, self.n_heads, self.dim_feedforward, self.dropout, batch_first=True, activation=self.activation_fn
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, self.n_layers)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="transformer")

    def forward(self, tracks, dets):
        tracks.tokens[tracks.masks] += self.track_encoder
        dets.tokens[dets.masks] += self.det_encoder

        assert not torch.any(torch.isnan(tracks.tokens))  # FIXME still nans from MB or reid?
        assert not torch.any(torch.isnan(dets.tokens))

        src = torch.cat(
            [dets.tokens, tracks.tokens, self.cls.repeat(dets.masks.shape[0], 1, 1)], dim=1
        )  # FIXME CLS not needed
        src = self.src_drop(self.src_norm(src))

        if self.src_key_padding_mask:
            src_mask = torch.cat(
                [
                    dets.masks,
                    tracks.masks,
                    torch.ones((dets.masks.shape[0], 1), device=dets.masks.device, dtype=torch.bool),
                ],
                dim=1,
            )
            x = self.encoder(src, src_key_padding_mask=~src_mask)
        else:
            src_mask = self.mask(dets.masks, tracks.masks)
            x = self.encoder(src, mask=src_mask)  # [B, D(+P) + T(+P) + 1, E]
        if self.use_processed_track_tokens:
            tracks.embs = x[:, dets.masks.shape[1]: dets.masks.shape[1] + tracks.masks.shape[1]]  # [B, T(+P), E]
        else:
            tracks.embs = tracks.tokens  # [B, T(+P), E]
        dets.embs = x[:, :dets.masks.shape[1]]  # [B, D(+P), E]
        return tracks, dets

    def mask(self, dets_masks, tracks_masks):
        """
        dets_masks: Tensor [B, D(+P)]
        tracks_masks: Tensor [B, T(+P)]

        returns: mask of shape [B * n_heads, D(+P)+T(+P)+1, D(+P)+T(+P)+1]
            padded values are set to True else is False
        """
        src_mask = torch.cat(
            [
                dets_masks,
                tracks_masks,
                torch.ones((dets_masks.shape[0], 1), device=dets_masks.device, dtype=torch.bool),
            ],
            dim=1,
        )

        src_mask = ~(src_mask.unsqueeze(2) * src_mask.unsqueeze(1))

        # just keep self-attention (otherwise becomes nan)
        indices = torch.arange(src_mask.shape[1])
        src_mask[:, indices, indices] = False

        # repeat for the n_heads
        src_mask = src_mask.repeat_interleave(self.n_heads, dim=0)
        return src_mask


class Decoder(Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 4096,
        dropout: int = 0.1,
        activation_fn: str = "gelu",
        checkpoint_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation_fn = activation_fn

        self.det_cls = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        self.track_cls = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        self.tgt_norm = nn.LayerNorm(emb_dim)
        self.tgt_drop = nn.Dropout(dropout)
        self.mem_norm = nn.LayerNorm(emb_dim)
        self.mem_drop = nn.Dropout(dropout)

        decoder_layers = nn.TransformerDecoderLayer(
            emb_dim, n_heads, dim_feedforward, dropout, batch_first=True, activation=activation_fn
        )
        self.decoder = nn.TransformerDecoder(decoder_layers, n_layers)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="transformer")

    def forward(self, tracks, dets):
        tgt = torch.cat([dets.tokens, self.det_cls.repeat(dets.masks.shape[0], 1, 1)], dim=1)
        tgt = self.tgt_drop(self.tgt_norm(tgt))
        mem = torch.cat([tracks.tokens, self.track_cls.repeat(tracks.masks.shape[0], 1, 1)], dim=1)
        mem = self.mem_drop(self.mem_norm(mem))

        tgt_mask, mem_mask = self.mask(dets.masks, tracks.masks)

        x = self.decoder(tgt, mem, tgt_mask, mem_mask)  # [B, D(+P)+1, E]
        dets.embs = x[:, : dets.masks.shape[1]]  # [B, D(+P), E]
        tracks.embs = tracks.tokens  # [B, T(+P), E]
        return tracks, dets

    def mask(self, dets_masks, tracks_masks):
        """
        dets_masks: Tensor [B, D(+P)]
        tracks_masks: Tensor [B, T(+P)]

        returns: 2 masks of shape [B * n_heads, D(+P)+1, D(+P)+1], [B * n_heads, D(+P)+1, T(+P)+1]
            padded values are set to True else is False
        """
        dets_masks = torch.cat(
            (
                dets_masks,
                torch.ones((dets_masks.shape[0], 1), device=dets_masks.device, dtype=torch.bool),
            ),
            dim=1,
        )
        tracks_masks = torch.cat(
            (
                tracks_masks,
                torch.ones((tracks_masks.shape[0], 1), device=dets_masks.device, dtype=torch.bool),
            ),
            dim=1,
        )
        tgt_mask = ~(dets_masks.unsqueeze(2) * dets_masks.unsqueeze(1))
        mem_mask = ~(dets_masks.unsqueeze(2) * tracks_masks.unsqueeze(1))

        # just keep self-attention (otherwise becomes nan)
        indices = torch.arange(tgt_mask.shape[1])
        tgt_mask[:, indices, indices] = False

        # to avoid becoming nan just put multi-head attention on cls token
        mem_mask[:, :, -1] = False

        # repeat for the n_heads
        tgt_mask = tgt_mask.repeat_interleave(self.n_heads, dim=0)
        mem_mask = mem_mask.repeat_interleave(self.n_heads, dim=0)

        return tgt_mask, mem_mask
