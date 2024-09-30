import logging
import time
from functools import partial

import numpy as np
import torch.nn as nn
import torch
from omegaconf import DictConfig
from .kalman_filter.ocsort_kalman_box_predictor import KalmanBoxTracker
from .motionbert.DSTformer import DSTformer, load_motion_bert_weights
from .motionbert.posetrack2h36m import posetrack2h36m
from .simformer_utils import (
    aggregate_dets_to_track,
    cosine_sim_matrix,
    hungarian_algorithm, iou_sim_matrix, euclidean_sim_matrix, display_bboxes,
)
from .simformer import SimFormer, Tokenizer

log = logging.getLogger(__name__)


def bbox_ltwh2ltrb(ltwh):  # TODO move to utils
    return np.concatenate((ltwh[:2], ltwh[:2] + ltwh[2:]))


def unnormalize_bbox(bbox, image_shape):
    return bbox * (list(image_shape) * 2)


def normalize_bbox(bbox, image_shape):
    return bbox / (list(image_shape) * 2)


def normalize_bbox_torch(bbox, image_shape):
    return bbox / torch.tensor(list(image_shape) * 2, dtype=torch.float32)


def ltrb_to_ltwh(bbox, rounded=False):
    """
    Converts coordinates `[left, top, right, bottom]` to `[left, top, w, h]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    bbox = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox


class KFTokenizer(nn.Module):
    def __init__(self, input_dim, n_hidden=0, hidden_dim=1024, emb_dim=1024):  # FIXME, use torch.MLP? Why multiple layers?
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        layers = []
        if n_hidden > 0:  # MLP
            layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(n_hidden):
                layers.extend([
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ])
            layers.extend([
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, emb_dim)
            ])
        else:  # linear projection
            layers.append(nn.Linear(input_dim, emb_dim))
        self.layers = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, x):
        if x.dim() == 2:
            x = self.layers(x)  # [batch_size, input_dim]
        elif x.dim() == 3:
            batch_size, S, input_dim = x.shape
            x = x.view(batch_size * S, input_dim)  # [batch_size * S, input_dim]
            x = self.layers(x)  # [batch_size * S, E]
            x = x.view(batch_size, S, self.emb_dim)  # [batch_size, S, E]
        else:
            raise ValueError(f"x should be of dim 2 or 3. Got: {x.dim()}")
        return x

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)


class DummyTokenizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class STSimFormer(SimFormer):
    """
    Uses only the foreground appearance embeddings to compute similarity.
    """
    def __init__(
        self,
        emb_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        train_cfg: DictConfig = None,
        checkpoint_path: str = None,
        mb_checkpoint_path: str = None,
        use_transformer: bool = True,
        use_kalman_filter: bool = False,
        use_tokenizer: bool = False,
        random_assignement: bool = False,
        debug_batch_display: bool = False,
        use_motion_bert: bool = False,
        dist_metric: bool = 'cosine',
        **kwargs,
    ):
        super().__init__(
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            train_cfg=train_cfg,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
        self.use_transformer = use_transformer
        self.use_kalman_filter = use_kalman_filter
        self.use_tokenizer = use_tokenizer
        self.random_assignement = random_assignement
        self.dist_metric = dist_metric
        self.total_time = 0
        self.debug_batch_display = debug_batch_display
        self.use_motion_bert = use_motion_bert
        self.best_threshold = None  # will be updated after last validation step
        if self.use_motion_bert:
            motion_bert = DSTformer(dim_feat=256, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.motion_bert = motion_bert
            # self.motion_bert = load_motion_bert_weights(motion_bert, mb_checkpoint_path)


    def tokenize(self, det_feats, dets_for_track_feats):  # FIXME should detections from tracklets be merged here? Do it in aggregate_dets_to_track? No because here tokenize must be done AFTER trakclet merging
        """
        inputs: dict of tensors
            age: st Tensor [B, N, 1]
            bbox_conf: st Tensor [B, N, 1]
            bbox_ltwh: st Tensor [B, N, 4]
            keypoints_xyc: st Tensor [B, N, 3*17]
            embeddings: app Tensor [B, N, 256*7]
            visibility_scores: app Tensor [B, N, 7]

        returns:
            tokens: Tensor [B, N, E]
        """
        # FIXME HANDLE CAMERA SWITCHES !!!!

        # model_backbone = DSTformer()
        # model_params = 0
        # for parameter in model_backbone.parameters():
        #     model_params = model_params + parameter.numel()
        # print('INFO: Trainable parameter count:', model_params)
        # chk_filename = os.path.join(opts.pretrained, opts.selection)
        # print('Loading checkpoint', chk_filename)
        # checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['model_pos'].items():
        #     name = k[7:]  # remove 'module.'
        #     new_state_dict[name] = v
        # model_backbone.load_state_dict(new_state_dict, strict=True)

        if self.use_motion_bert:
            # TODO :
            #   adapt inference !!!!!!!
            #   take into account tracklets_age !!!!!!!!!!!!
            detections_kp_xyc = self.motionbert_preprocess(det_feats["keypoints_xyc"])
            tracklets_kp_xyc = self.motionbert_preprocess(dets_for_track_feats["keypoints_xyc"])  # TODO tracklets_kp_xyc in god order? left to right?
            tracklets_features = self.motion_bert.get_representation(tracklets_kp_xyc)  # [B, T, 17, D=512]
            detections_features = self.motion_bert.get_representation(detections_kp_xyc)  # [B, T, 17, D=512]
            tracklets_features = tracklets_features.mean(dim=(1, 2)).unsqueeze(1)  # [B, D=512]
            detections_features = detections_features.mean(dim=(1, 2)).unsqueeze(1)  # [B, D=512]
        elif self.use_kalman_filter:
            standard_img_shape = [1920, 1080]  # FIXME
            start_time = time.time()
            tracklets_bbox_ltwh = dets_for_track_feats['bbox_ltwh'].cpu().numpy()
            dets_bbox_ltwh = det_feats['bbox_ltwh']
            tracklets_age = dets_for_track_feats['age'].squeeze(-1).cpu().numpy()

            predictions = []
            for t_idx, tracklet_bbox_ltwh in enumerate(tracklets_bbox_ltwh):
                bbox_ltwh = tracklet_bbox_ltwh[0]
                bbox_ltrb = bbox_ltwh2ltrb(bbox_ltwh)
                bbox_ltrb = unnormalize_bbox(bbox_ltrb, standard_img_shape)
                tracklet_kf = KalmanBoxTracker(bbox_ltrb)
                detections_per_age = {age: bbox_ltwh for age, bbox_ltwh in zip(tracklets_age[t_idx, 1:], tracklet_bbox_ltwh[1:]) if age > 0}
                oldest = int(tracklets_age[t_idx, 0])
                for age in range(oldest, 0, -1):
                    if age in detections_per_age:
                        bbox_ltwh = detections_per_age[age]
                        bbox_ltrb = bbox_ltwh2ltrb(bbox_ltwh)
                        bbox_ltrb = unnormalize_bbox(bbox_ltrb, standard_img_shape)
                        tracklet_kf.update(bbox_ltrb)
                    else:
                        tracklet_kf.update(None)
                predicted_bbox_ltrb = tracklet_kf.predict()[0]
                if np.any(np.isnan(predicted_bbox_ltrb)):
                    predicted_bbox_ltrb = tracklet_kf.history_observations[-1]
                predicted_bbox_ltrb = normalize_bbox(predicted_bbox_ltrb, standard_img_shape)
                predicted_bbox_ltwh = ltrb_to_ltwh(predicted_bbox_ltrb)
                predictions.append(predicted_bbox_ltwh)
            predictions = np.stack(predictions, axis=0)
            tracks_pred_bbox_ltwh = torch.from_numpy(predictions).to(det_feats['bbox_ltwh'].device)
            tracks_pred_bbox_ltwh = tracks_pred_bbox_ltwh.unsqueeze(1)
            detections_features = dets_bbox_ltwh.to(det_feats['bbox_ltwh'].dtype)
            tracklets_features = tracks_pred_bbox_ltwh.to(det_feats['bbox_ltwh'].dtype)
            exec_time = time.time() - start_time
            self.total_time += exec_time
            # print(f"Total time: {self.total_time} s")
        else:
            tracklets_bbox_ltwh = dets_for_track_feats['bbox_ltwh'][:, 1:]
            tracklets_bbox_ltwh = tracklets_bbox_ltwh[:, :1]
            dets_bbox_ltwh = det_feats['bbox_ltwh']
            detections_features = dets_bbox_ltwh
            tracklets_features = tracklets_bbox_ltwh

        if self.use_tokenizer:
            detections_features = self.st_tokenizer(detections_features)  # TODO requires grad
            tracklets_features = self.st_tokenizer(tracklets_features)  # TODO requires grad

        return detections_features, tracklets_features

    def motionbert_preprocess(self, keypoints_xyc):
        keypoints_xyc = keypoints_xyc.clone().view(*keypoints_xyc.shape[:-1], 17, 3)
        keypoints_xyc[..., :2] = keypoints_xyc[..., :2] * 2 - 1  # !!! flag = -1 -> use nans? before normalization?
        keypoints_xyc = posetrack2h36m(keypoints_xyc)
        keypoints_xyc[keypoints_xyc[:, :, :, 2] == 0] = 0  # for invisible keypoints
        keypoints_xyc[keypoints_xyc[:, :, :, 2] == -1] = 0  # for not available skeletons
        return keypoints_xyc

    def preprocess_train_val_batch(self, batch, B, N):
        """
        batch: dict of tensors
            det_feats: dict with tensors of features (B*N, 1, F)
            det_targets: dict with tensors of targets - track_ids (B*N, 1)
            dets_for_track_feats: dict with tensors of features (B*N, T(+P), F)
            dets_for_track_targets: dict with tensors of targets - track_ids (B*N, T(+P), 1)

        returns: tuple
            det_tokens: Tensor [B, N, E]
            det_masks: Tensor [B, N]
            det_targets: Tensor [B, N]
            track_tokens: Tensor [B, N, E]
            track_masks: Tensor [B, N]
            track_targets: Tensor [B, N]
        """
        det_feats = batch["det_feats"]
        det_targets = batch["det_targets"]
        dets_for_track_feats = batch["dets_for_track_feats"]
        dets_for_track_targets = batch["dets_for_track_targets"]
        if 'image_path' in batch:
            image_paths = np.array(batch['image_path'])
        else:
            image_paths = None
        det_masks = torch.where(det_targets == -1, False, True)  # FIXME quid

        det_tokens, dets_for_track_tokens = self.tokenize(det_feats, dets_for_track_feats)

        track_tokens, track_masks, track_targets = aggregate_dets_to_track(dets_for_track_tokens, dets_for_track_targets)

        if B is not None:
            if det_tokens.shape[0] != N * B:  # fixme this is ugly, force dataloader to work with multiples of B*N samples
                B = max(1, det_tokens.shape[0] // N)
                N = det_tokens.shape[0] // B
                det_tokens = det_tokens[:B * N]
                track_tokens = track_tokens[:B * N]
                det_masks = det_masks[:B * N]
                track_masks = track_masks[:B * N]
                det_targets = det_targets[:B * N]
                track_targets = track_targets[:B * N]
                image_paths = image_paths[:B * N]
            det_tokens = det_tokens.reshape((B, N, -1))
            track_tokens = track_tokens.reshape((B, N, -1))
            det_masks = det_masks.reshape((B, N))
            track_masks = track_masks.reshape((B, N))
            det_targets = det_targets.reshape((B, N))
            track_targets = track_targets.reshape((B, N))
            image_paths = image_paths.reshape((B, N))
        else:
            track_tokens = track_tokens.unsqueeze(0)
            track_masks = track_masks.unsqueeze(0)
            track_targets = track_targets.unsqueeze(0)
            det_tokens = det_tokens.unsqueeze(0).squeeze(2)
            det_masks = det_masks.unsqueeze(0).squeeze(1)
            det_targets = det_targets.unsqueeze(0).squeeze(1)
        return det_tokens, det_masks, det_targets, track_tokens, track_masks, track_targets

    def forward(self, det_tokens, det_masks, track_tokens, track_masks, ret_sim_matrix=False):
        """
        det_tokens: Tensor [B, D(+P), E]
        det_masks: Tensor [B, D(+P)]
            True if det
            False if pad
        track_tokens: Tensor [B, T(+P), E]
        track_masks: Tensor [B, T(+P)]
            True if det
            False if pad
        ret_sim_matrix: bool

        returns: tuple
            det_outputs [B, D(+P), E]
            track_outputs [B, T(+P), E]
            (opt) td_sim_matrix [B, T(+P), D(+P)]
                  padded pairs are set to -inf
        """
        if self.random_assignement:
            det_embs = det_tokens
            track_embs = track_tokens
            if ret_sim_matrix:
                td_sim_matrix = torch.rand((det_embs.shape[0], track_embs.shape[1], det_embs.shape[1]), device=det_embs.device)
                return det_embs, track_embs, td_sim_matrix
            else:
                return det_embs, track_embs
        elif not self.use_transformer:
            det_embs = det_tokens
            track_embs = track_tokens
            if ret_sim_matrix:
                if self.use_tokenizer and not isinstance(self.st_tokenizer, DummyTokenizer) or self.use_motion_bert:
                    if self.dist_metric == 'cosine':
                        td_sim_matrix = cosine_sim_matrix(track_embs, track_masks, det_embs, det_masks)
                    elif self.dist_metric == 'euclidean':
                        td_sim_matrix = euclidean_sim_matrix(track_embs, track_masks, det_embs, det_masks)
                    else:
                        raise ValueError(f"Unknown dist_metric: {self.dist_metric}")
                else:
                    td_sim_matrix = iou_sim_matrix(track_embs, track_masks, det_embs, det_masks)
                return det_embs, track_embs, td_sim_matrix
            return det_embs, track_embs
        else:
            return SimFormer.forward(self, det_tokens, det_masks, track_tokens, track_masks, ret_sim_matrix)
