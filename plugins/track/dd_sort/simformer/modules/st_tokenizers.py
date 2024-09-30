from functools import partial
from typing import List

import torch
import torch.nn as nn

from torchvision.ops import MLP
from .transformers import Module
from ..motionbert.DSTformer import DSTformer, load_motion_bert_weights
from ..motionbert.posetrack2h36m import posetrack2h36m


class MotionBertTokenizer(Module):
    default_similarity_metric = "cosine"

    def __init__(self, token_dim: int, mb_checkpoint_path, tracklet_max_age, pad_empty_frames, enable_ll=True, freeze=False, checkpoint_path: str = None, **kwargs):
        super().__init__()
        motion_bert = DSTformer(dim_feat=256, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.dim_mb_out_feat = 512
        self.motion_bert = load_motion_bert_weights(motion_bert, mb_checkpoint_path)
        self.token_dim = token_dim
        self.freeze = freeze
        self.tracklet_max_age = tracklet_max_age
        self.pad_empty_frames = pad_empty_frames

        self.enable_ll = enable_ll
        self.linear = nn.Linear(self.dim_mb_out_feat, self.token_dim)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.MotionBertTokenizer")

        if self.freeze:
            for param in self.motion_bert.parameters():
                param.requires_grad = False

    def forward(self, tracklets):
        tracklets_kp_xyc = self.motionbert_preprocess(tracklets.feats['keypoints_xyc'], tracklets.feats['age'], self.tracklet_max_age, self.pad_empty_frames)
        tracklets_embeds = self.motion_bert.get_representation(tracklets_kp_xyc.flatten(0, 1))  # [B, T, 17, D=512]
        tracklets_embeds = tracklets_embeds.mean(dim=(1, 2))  # [B, D=512]
        tracklets_embeds = tracklets_embeds.unflatten(0, tracklets_kp_xyc.shape[:2])
        if self.enable_ll:
            tracklets_embeds = self.linear(tracklets_embeds)
        return tracklets_embeds

    @staticmethod
    def motionbert_preprocess(keypoints_xyc, ages, max_age, pad_empty_frames=True):
        keypoints_xyc = keypoints_xyc.clone()
        keypoints_xyc[..., :2] = keypoints_xyc[..., :2] * 2 - 1  # normalization to 0->1 range !!! flag = -1 -> use nans? before normalization?
        B, N = keypoints_xyc.shape[:2]
        keypoints_xyc = posetrack2h36m(keypoints_xyc.flatten(0, 1))
        keypoints_xyc = keypoints_xyc.unflatten(0, (B, N))
        keypoints_xyc[keypoints_xyc[..., 2] == 0] = 0  # for invisible keypoints
        if pad_empty_frames:
            keypoints_xyc = extract_keypoints_by_age(keypoints_xyc, ages.squeeze(-1), max_age)
        else:
            keypoints_xyc[keypoints_xyc[..., 2].isnan()] = 0  # for not available skeletons
        keypoints_xyc = keypoints_xyc.flip(dims=[2])  # reverse detections order, MotionBERT format = index 0 as oldest detection
        return keypoints_xyc


def batch_extract_keypoints_by_age(keypoints_xyc, ages, max_age):
    B, N = ages.shape[:2]

    valid_mask = (~torch.isnan(ages)) & (ages < max_age)

    # Construct an array of indices for where to place the keypoints in the result tensor
    ages_indices = torch.zeros_like(ages, dtype=torch.long, device=keypoints_xyc.device)
    ages_indices[valid_mask] = ages[valid_mask].long()

    # Initialize the result tensor
    padded_keypoints = torch.zeros(B, N, max_age, 17, 3, device=keypoints_xyc.device)

    # Assign the valid keypoints to the correct position in the result tensor
    for b in range(B):
        for n in range(N):
            valid_positions = ages_indices[b, n][valid_mask[b, n]]
            padded_keypoints[b, n, valid_positions] = keypoints_xyc[b, n][valid_mask[b, n]]

    return padded_keypoints


def extract_keypoints_by_age(b_keypoints, b_ages, max_age):
    try:
        B, N = b_ages.shape[:2]
        b_padded_keypoints = torch.zeros(B, N, max_age, 17, 3, device=b_keypoints.device)
        for b in range(B):
            for n in range(N):
                keypoints = b_keypoints[b, n][~torch.isnan(b_ages[b, n])]
                ages = b_ages[b, n][~torch.isnan(b_ages[b, n])]
                keypoints = keypoints[ages < max_age]
                ages = ages[ages < max_age]
                padded_keypoints = torch.zeros(max_age, 17, 3, device=b_keypoints.device)
                padded_keypoints[ages.long()] = keypoints
                b_padded_keypoints[b, n] = padded_keypoints
    except:
        print("An exception occurred")
    return b_padded_keypoints


class LastBboxTokenizer(Module):
    """
    The last bbox of a tracklet will be directly encoded with the MLP to produce the output token.
    """

    def __init__(self, feat_dim: int, token_dim: int, hidden_channels: List[int], freeze=False, checkpoint_path: str = None, no_mlp = False, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self.token_dim = token_dim
        self.hidden_channels = hidden_channels
        self.freeze = freeze
        self.no_mlp = no_mlp
        self.default_similarity_metric = "iou" if self.no_mlp else "cosine"
        self.mlp = MLP(
            in_channels=self.feat_dim,
            hidden_channels=self.hidden_channels+[self.token_dim],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=True,
        )

        self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.MotionBertTokenizer")

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, tracklets):
        last_bboxes = tracklets.feats['bbox_ltwh'][:, :, 0]
        if self.no_mlp:
            return last_bboxes
        valid_last_bboxes_mask = tracklets.feats_masks[..., 0].flatten()
        valid_last_bboxes = last_bboxes.flatten(0, 1)[valid_last_bboxes_mask]
        valid_bbox_features = self.mlp(valid_last_bboxes)
        bbox_features = torch.zeros((len(valid_last_bboxes_mask), self.token_dim), device=valid_bbox_features.device)
        bbox_features[valid_last_bboxes_mask] = valid_bbox_features
        bbox_features = bbox_features.unflatten(0, tracklets.feats_masks[..., 0].shape)
        return bbox_features


# class KFTokenizer(Module):  # TODO
#     """
#     The last bbox of a tracklet will be directly encoded with the MLP to produce the output token.
#     """
#     default_similarity_metric = "cosine"
# 
#     def __init__(self, feat_dim: int, token_dim: int, hidden_channels: List[int], freeze=False, checkpoint_path: str = None, **kwargs):
#         super().__init__()
#         self.feat_dim = feat_dim
#         self.token_dim = token_dim
#         self.hidden_channels = hidden_channels
#         self.freeze = freeze
#         self.mlp = MLP(
#             in_channels=self.feat_dim,
#             hidden_channels=self.hidden_channels+[self.token_dim],
#             norm_layer=nn.BatchNorm1d,
#             activation_layer=nn.ReLU,
#             bias=True,
#         )
# 
#         self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.MotionBertTokenizer")
# 
#         if self.freeze:
#             for param in self.parameters():
#                 param.requires_grad = False
# 
# 
#     def forward(self, tracklets):
#         last_bboxes = tracklets.feats['bbox_ltwh'][:, :, 0]
#         valid_last_bboxes_mask = tracklets.feats_masks[..., 0].flatten()
#         valid_last_bboxes = last_bboxes.flatten(0, 1)[valid_last_bboxes_mask]
#         standard_img_shape = [1920, 1080]  # FIXME
#         start_time = time.time()
#         tracklets_bbox_ltwh = dets_for_track_feats['bbox_ltwh'].cpu().numpy()
#         dets_bbox_ltwh = det_feats['bbox_ltwh']
#         tracklets_age = dets_for_track_feats['age'].squeeze(-1).cpu().numpy()
# 
#         predictions = []
#         for t_idx, tracklet_bbox_ltwh in enumerate(tracklets_bbox_ltwh):
#             bbox_ltwh = tracklet_bbox_ltwh[0]
#             bbox_ltrb = bbox_ltwh2ltrb(bbox_ltwh)
#             bbox_ltrb = unnormalize_bbox(bbox_ltrb, standard_img_shape)
#             tracklet_kf = KalmanBoxTracker(bbox_ltrb)
#             detections_per_age = {age: bbox_ltwh for age, bbox_ltwh in
#                                   zip(tracklets_age[t_idx, 1:], tracklet_bbox_ltwh[1:]) if age > 0}
#             oldest = int(tracklets_age[t_idx, 0])
#             for age in range(oldest, 0, -1):
#                 if age in detections_per_age:
#                     bbox_ltwh = detections_per_age[age]
#                     bbox_ltrb = bbox_ltwh2ltrb(bbox_ltwh)
#                     bbox_ltrb = unnormalize_bbox(bbox_ltrb, standard_img_shape)
#                     tracklet_kf.update(bbox_ltrb)
#                 else:
#                     tracklet_kf.update(None)
#             predicted_bbox_ltrb = tracklet_kf.predict()[0]
#             if np.any(np.isnan(predicted_bbox_ltrb)):
#                 predicted_bbox_ltrb = tracklet_kf.history_observations[-1]
#             predicted_bbox_ltrb = normalize_bbox(predicted_bbox_ltrb, standard_img_shape)
#             predicted_bbox_ltwh = ltrb_to_ltwh(predicted_bbox_ltrb)
#             predictions.append(predicted_bbox_ltwh)
#         predictions = np.stack(predictions, axis=0)
#         tracks_pred_bbox_ltwh = torch.from_numpy(predictions).to(det_feats['bbox_ltwh'].device)
#         tracks_pred_bbox_ltwh = tracks_pred_bbox_ltwh.unsqueeze(1)
#         detections_features = dets_bbox_ltwh.to(det_feats['bbox_ltwh'].dtype)
#         tracklets_features = tracks_pred_bbox_ltwh.to(det_feats['bbox_ltwh'].dtype)
#         exec_time = time.time() - start_time
#         self.total_time += exec_time


#
# def tokenize(self, det_feats, track_feats):  # FIXME should detections from tracklets be merged here? Do it in aggregate_dets_to_track? No because here tokenize must be done AFTER trakclet merging
#     """
#     inputs: dict of tensors
#         age: st Tensor [B, N, 1]
#         bbox_conf: st Tensor [B, N, 1]
#         bbox_ltwh: st Tensor [B, N, 4]
#         keypoints_xyc: st Tensor [B, N, 3*17]
#         embeddings: app Tensor [B, N, 256*7]
#         visibility_scores: app Tensor [B, N, 7]
#
#     returns:
#         tokens: Tensor [B, N, E]
#     """
#     # FIXME HANDLE CAMERA SWITCHES !!!!
#
#     # model_backbone = DSTformer()
#     # model_params = 0
#     # for parameter in model_backbone.parameters():
#     #     model_params = model_params + parameter.numel()
#     # print('INFO: Trainable parameter count:', model_params)
#     # chk_filename = os.path.join(opts.pretrained, opts.selection)
#     # print('Loading checkpoint', chk_filename)
#     # checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
#     # new_state_dict = OrderedDict()
#     # for k, v in checkpoint['model_pos'].items():
#     #     name = k[7:]  # remove 'module.'
#     #     new_state_dict[name] = v
#     # model_backbone.load_state_dict(new_state_dict, strict=True)
#
#     if self.use_motion_bert:
#         # TODO :
#         #   adapt inference !!!!!!!
#         #   take into account tracklets_age !!!!!!!!!!!!
#         detections_kp_xyc = self.motionbert_preprocess(det_feats["keypoints_xyc"])
#         tracklets_kp_xyc = self.motionbert_preprocess(track_feats["keypoints_xyc"])  # TODO tracklets_kp_xyc in god order? left to right?
#         tracklets_features = self.motion_bert.get_representation(tracklets_kp_xyc)  # [B, T, 17, D=512]
#         detections_features = self.motion_bert.get_representation(detections_kp_xyc)  # [B, T, 17, D=512]
#         tracklets_features = tracklets_features.mean(dim=(1, 2)).unsqueeze(1)  # [B, D=512]
#         detections_features = detections_features.mean(dim=(1, 2)).unsqueeze(1)  # [B, D=512]
#     elif self.use_kalman_filter:
#         standard_img_shape = [1920, 1080]  # FIXME
#         start_time = time.time()
#         tracklets_bbox_ltwh = track_feats['bbox_ltwh'].cpu().numpy()
#         dets_bbox_ltwh = det_feats['bbox_ltwh']
#         tracklets_age = track_feats['age'].squeeze(-1).cpu().numpy()
#
#         predictions = []
#         for t_idx, tracklet_bbox_ltwh in enumerate(tracklets_bbox_ltwh):
#             bbox_ltwh = tracklet_bbox_ltwh[0]
#             bbox_ltrb = bbox_ltwh2ltrb(bbox_ltwh)
#             bbox_ltrb = unnormalize_bbox(bbox_ltrb, standard_img_shape)
#             tracklet_kf = KalmanBoxTracker(bbox_ltrb)
#             detections_per_age = {age: bbox_ltwh for age, bbox_ltwh in zip(tracklets_age[t_idx, 1:], tracklet_bbox_ltwh[1:]) if age > 0}
#             oldest = int(tracklets_age[t_idx, 0])
#             for age in range(oldest, 0, -1):
#                 if age in detections_per_age:
#                     bbox_ltwh = detections_per_age[age]
#                     bbox_ltrb = bbox_ltwh2ltrb(bbox_ltwh)
#                     bbox_ltrb = unnormalize_bbox(bbox_ltrb, standard_img_shape)
#                     tracklet_kf.update(bbox_ltrb)
#                 else:
#                     tracklet_kf.update(None)
#             predicted_bbox_ltrb = tracklet_kf.predict()[0]
#             if np.any(np.isnan(predicted_bbox_ltrb)):
#                 predicted_bbox_ltrb = tracklet_kf.history_observations[-1]
#             predicted_bbox_ltrb = normalize_bbox(predicted_bbox_ltrb, standard_img_shape)
#             predicted_bbox_ltwh = ltrb_to_ltwh(predicted_bbox_ltrb)
#             predictions.append(predicted_bbox_ltwh)
#         predictions = np.stack(predictions, axis=0)
#         tracks_pred_bbox_ltwh = torch.from_numpy(predictions).to(det_feats['bbox_ltwh'].device)
#         tracks_pred_bbox_ltwh = tracks_pred_bbox_ltwh.unsqueeze(1)
#         detections_features = dets_bbox_ltwh.to(det_feats['bbox_ltwh'].dtype)
#         tracklets_features = tracks_pred_bbox_ltwh.to(det_feats['bbox_ltwh'].dtype)
#         exec_time = time.time() - start_time
#         self.total_time += exec_time
#         # print(f"Total time: {self.total_time} s")
#     else:
#         tracklets_bbox_ltwh = track_feats['bbox_ltwh'][:, 1:]
#         tracklets_bbox_ltwh = tracklets_bbox_ltwh[:, :1]
#         dets_bbox_ltwh = det_feats['bbox_ltwh']
#         detections_features = dets_bbox_ltwh
#         tracklets_features = tracklets_bbox_ltwh
#
#     if self.use_tokenizer:
#         detections_features = self.st_tokenizer(detections_features)  # TODO requires grad
#         tracklets_features = self.st_tokenizer(tracklets_features)  # TODO requires grad
#
#     return detections_features, tracklets_features
#
# def motionbert_preprocess(self, keypoints_xyc):
#     keypoints_xyc = keypoints_xyc.clone().view(*keypoints_xyc.shape[:-1], 17, 3)
#     keypoints_xyc[..., :2] = keypoints_xyc[..., :2] * 2 - 1  # !!! flag = -1 -> use nans? before normalization?
#     keypoints_xyc = posetrack2h36m(keypoints_xyc)
#     keypoints_xyc[keypoints_xyc[:, :, :, 2] == 0] = 0  # for invisible keypoints
#     keypoints_xyc[keypoints_xyc[:, :, :, 2] == -1] = 0  # for not available skeletons
#     return keypoints_xyc
