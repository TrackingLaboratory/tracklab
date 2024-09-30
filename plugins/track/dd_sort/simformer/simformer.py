import logging
from dataclasses import dataclass

import pytorch_lightning as pl
import torch.nn as nn
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_metric_learning import distances, losses, miners, reducers

from .assignement_strats import hungarian_algorithm, argmax_algorithm
from .callbacks import SimMetrics, ClsMetrics
from .merge_token_strats import merge_token_strats
from .similarity_metrics import similarity_metrics
from .utils import *

log = logging.getLogger(__name__)


@dataclass
class Tracklets:
    def __init__(self, features, feats_masks, targets=None):
        """
        :param feats: dict of tensors float32 [B, N, T, F]
        :param feats_masks: tensor bool [B, N, T]
        :param masks: tensor bool [B, N]
        :param tokens: tensor float32 [B, N, E]
        :param embs: tensor float32 [B, N, E]
        :param targets: tensor float32 [B, N]
        """
        self.feats = features
        self.feats_masks = feats_masks
        self.masks = self.feats_masks.any(dim=-1)
        self.tokens = None
        self.embs = None
        if targets is not None and len(targets.shape) > 2:
            self.targets = targets[:, :, 0]
        else:
            self.targets = targets


@dataclass
class Detections(Tracklets):
    def __init__(self, features, feats_masks, targets=None):
        """
        :param feats: dict of tensors float32 [B, N, 1, F]
        :param feats_masks: tensor bool [B, N, 1]
        :param masks: tensor bool [B, N]
        :param tokens: tensor float32 [B, N, E]
        :param embs: tensor float32 [B, N, E]
        :param targets: tensor float32 [B, N]
        """
        assert feats_masks.shape[2] == 1
        super().__init__(features, feats_masks, targets)


class SimFormer(pl.LightningModule):
    def __init__(
            self,
            transformer_cfg: DictConfig,
            tokenizers_cfg: DictConfig,
            classifier_cfg: DictConfig = None,
            train_cfg: DictConfig = None,
            batch_transforms: DictConfig = None,
            merge_token_strat: str = "sum",
            sim_strat: str = "cosine",
            assos_strat: str = "hungarian",
            sim_threshold: int = 0.7,
            tl_margin: float = 0.3,
            loss_strat: str = "triplet",
            contrastive_loss_strat: str = "inter_intra",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = instantiate(transformer_cfg)
        self.tokenizers = nn.ModuleDict({n: instantiate(t) for n, t in tokenizers_cfg.items() if t['_enabled_']})
        self.train_cfg = train_cfg
        self.merge_token_strat = merge_token_strat
        self.sim_strat = sim_strat
        self.assos_strat = assos_strat
        self.computed_sim_threshold = None
        self.computed_sim_threshold2 = None
        self.best_distr_overlap_threshold = None
        self.sim_threshold = sim_threshold
        self.final_tracking_threshold = sim_threshold
        log.info(f"SimFormer initialized with final_tracking_threshold={sim_threshold} from yaml config. Will be overwritten later by an optimized threshold if SimFormer validation is enabled.")
        self.tl_margin = tl_margin
        self.loss_strat = loss_strat
        self.contrastive_loss_strat = contrastive_loss_strat

        # handle instantiation functions
        if batch_transforms is not None:
            self.batch_transforms = {n: instantiate(t) for n, t in batch_transforms.items()}
        else:
            self.batch_transforms = {}

        # merging
        if merge_token_strat in merge_token_strats:
            self.merge = merge_token_strats[merge_token_strat]
        else:
            raise NotImplementedError

        # similarity
        if sim_strat in similarity_metrics:
            self.similarity_metric = similarity_metrics[sim_strat]
        else:
            raise NotImplementedError

        # association
        if assos_strat == "hungarian":
            self.association = hungarian_algorithm
        elif assos_strat == "argmax":
            self.association = argmax_algorithm
        else:
            raise NotImplementedError

        # loss on sim
        distance = distances.CosineSimilarity()
        reducer = reducers.AvgNonZeroReducer()
        if self.loss_strat == "triplet":
            self.emb_mining = miners.TripletMarginMiner(
                margin=self.tl_margin, distance=distance, type_of_triplets="all"
            )
            self.sim_loss = losses.TripletMarginLoss(margin=self.tl_margin, distance=distance, reducer=reducer)
        elif self.loss_strat == "infoNCE":
            self.sim_loss = losses.NTXentLoss(distance=distance, reducer=reducer)
        else:
            raise NotImplementedError

        # classifier
        if classifier_cfg is not None and classifier_cfg._enabled_:
            self.classifier = instantiate(classifier_cfg)
            self.alpha_loss = train_cfg.alpha_loss

            self.sigmoid = nn.functional.sigmoid
            self.bce = nn.functional.binary_cross_entropy_with_logits
        else:
            self.alpha_loss = 1.0

    def training_step(self, batch, batch_idx):
        if "train" in self.batch_transforms:
            batch = self.batch_transforms["train"](batch)

        tracks, dets = self.train_val_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        sim_loss, cls_loss = self.compute_loss(tracks, dets, td_sim_matrix)

        loss = self.log_loss(cls_loss, sim_loss, "train")
        return {
            "loss": loss,
            "dets": dets,
            "tracks": tracks,
            "td_sim_matrix": td_sim_matrix,
        }

    def validation_step(self, batch, batch_idx):
        if "val" in self.batch_transforms:
            batch = self.batch_transforms["val"](batch)
        tracks, dets = self.train_val_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        sim_loss, cls_loss = self.compute_loss(tracks, dets, td_sim_matrix)

        loss = self.log_loss(cls_loss, sim_loss, "val")
        return {
            "loss": loss,
            "tracks": tracks,
            "dets": dets,
            "td_sim_matrix": td_sim_matrix,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        tracks, dets = self.predict_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        # best_cls_roc_threshold = self.best_roc_cls_threshold if self.best_roc_cls_threshold else self.det_threshold  # fixme
        association_matrix, association_result = self.association(td_sim_matrix, tracks.masks, dets.masks,
                                                                  sim_threshold=self.final_tracking_threshold)
        # plt = display_bboxes(tracks, dets, None, batch["images"])
        # plt.show()
        return association_matrix, association_result, td_sim_matrix

    def forward(self, tracks, dets):
        tracks, dets = self.tokenize(tracks, dets)  # feats -> list(tokens)
        tracks, dets = self.merge(tracks, dets)  # list(tokens) -> tokens
        tracks, dets = self.transformer(tracks, dets)  # tokens -> embs
        if hasattr(self, "classifier"):
            dets = self.classifier(dets)

        td_sim_matrix = self.similarity(tracks, dets)  # embs -> sim_matrix

        # print("SIMFORMER")
        # print("DETECTIONS")
        # print([c for c in dets.feats['bbox_conf'][0, :, 0, 0].cpu().numpy()])
        # print([c for c in dets.feats['index'][0, :, 0].cpu().numpy()])
        # print([c for c in dets.feats['bbox_ltwh'][0, :, 0, 0].cpu().numpy()])
        # print([c for c in dets.feats['embeddings'][0, :, 0, 1].cpu().numpy()])
        # print("TRACKS")
        # print([c for c in tracks.feats['bbox_conf'][0, :, 0, 0].cpu().numpy()])
        # print([c for c in tracks.feats['index'][0, :, 0].cpu().numpy()])
        # print([c for c in tracks.feats['bbox_ltwh'][0, :, 0, 0].cpu().numpy()])
        # print([c for c in tracks.feats['embeddings'][0, :, 0, 1].cpu().numpy()])
        # print("COST MATRIX")
        # print(td_sim_matrix.cpu().numpy())
        return tracks, dets, td_sim_matrix

    def train_val_preprocess(self, batch):  # TODO merge with predict_preprocess, compute det/trask masks in getitem
        """
        :param batch:
            dict of tensors containing the inputs features and targets of detections and tracklets
        :return:
            dets: Detections - a dataclass wrapper containing batch infos for detections
            tracks: Tracklets - a dataclass wrapper containing batch infos for tracklets
        """

        tracks = Tracklets(batch["track_feats"], ~batch["track_targets"].isnan(), batch["track_targets"])
        dets = Detections(batch["det_feats"], ~batch["det_targets"].isnan(), batch["det_targets"])
        return tracks, dets

    def predict_preprocess(self, batch):
        """
        :param batch:
            dict of tensors containing the inputs features and targets of detections and tracklets
        :return:
            dets: Detections - a dataclass wrapper containing batch infos for detections
            tracks: Tracklets - a dataclass wrapper containing batch infos for tracklets
        """

        tracks = Tracklets(batch["track_feats"], batch["track_masks"])
        dets = Detections(batch["det_feats"], batch["det_masks"])
        return tracks, dets

    def tokenize(self, tracks, dets):
        """
        Operate the tokenization step for the differents tokenizers
        :param dets: Detections
        :param tracks: Tracklets
        :return: updated dets and tracks with partial tokens in a dict not merged
        """
        tracks.tokens = {}
        dets.tokens = {}
        for n, t in self.tokenizers.items():
            tracks.tokens[n] = t(tracks)
            dets.tokens[n] = t(dets)
        return tracks, dets

    def similarity(self, tracks, dets):
        """
        Compute the similarity matrix between the tokens of dets and tracks.
        if tokens is a list of N tensors and not a single tensor, N similarity matrices are computed and averaged
        """
        # FIXME similarity_metric should be a list, a different metric could be used for each type of token
        if isinstance(tracks.embs, dict):
            td_sim_matrix = []
            for (tokenizer_name, t), (_, d) in zip(tracks.embs.items(), dets.embs.items()):
                if self.sim_strat == "default_for_each_token_type":
                    sm = similarity_metrics[self.tokenizers[tokenizer_name].default_similarity_metric]
                    td_sim_matrix.append(sm(t, tracks.masks, d, dets.masks))
                else:
                    td_sim_matrix.append(self.similarity_metric(t, tracks.masks, d, dets.masks))
            td_sim_matrix = torch.stack(td_sim_matrix).mean(dim=0)
        else:
            td_sim_matrix = self.similarity_metric(tracks.embs, tracks.masks, dets.embs, dets.masks)
        return td_sim_matrix

    def compute_loss(self, tracks, dets, *args):
        """
        :param dets: dataclass
            embs tensor float32 dim [B, D, E]
            confs tensor float32 dim [B, D]
            masks tensor bool dim [B, D]
            targets tensor float32 dim [B, D]
        :param tracks: dataclass
            embs tensor float32 dim [B, T, E]
            masks tensor bool dim [B, T]
            targets tensor float32 dim [B, T]
        :param td_sim_matrix: unused
        :return: sim_loss float32 and cls_loss float32
        """
        if not isinstance(tracks.embs, dict):
            tracks.embs = {"default": tracks.embs}
            dets.embs = {"default": dets.embs}

        n_tokens = len(tracks.embs.keys())
        B = list(tracks.embs.values())[0].shape[0]

        # Initialize loss variables
        sim_loss = torch.zeros((n_tokens, B), dtype=torch.float32, device=self.device)
        cls_loss = torch.zeros((n_tokens, B), dtype=torch.float32, device=self.device)
        mask_sim_loss = torch.zeros((n_tokens, B), dtype=torch.bool, device=self.device)
        mask_cls_loss = torch.zeros((n_tokens, B), dtype=torch.bool, device=self.device)

        for h, token_name in enumerate(tracks.embs.keys()):
            tracks_embs = tracks.embs[token_name]
            dets_embs = dets.embs[token_name]

            for i in range(B):
                masked_track_embs = tracks_embs[i, tracks.masks[i]]
                masked_track_targets = tracks.targets[i, tracks.masks[i]]
                masked_det_embs = dets_embs[i, dets.masks[i]]
                masked_det_targets = dets.targets[i, dets.masks[i]]

                if len(masked_det_embs) != 0 or len(masked_track_embs) != 0:
                    mask_sim_loss[h, i] = True

                    if self.loss_strat == "triplet":
                        if self.contrastive_loss_strat == "inter_intra":
                            # Compute embeddings loss on all tracks/detections (track_ids != 0)
                            embeddings = torch.cat([masked_track_embs, masked_det_embs], dim=0)
                            labels = torch.cat([masked_track_targets, masked_det_targets], dim=0)
                            indices_tuple = self.emb_mining(embeddings, labels)
                            sim_loss[h, i] = self.sim_loss(embeddings, labels, indices_tuple)
                        elif self.contrastive_loss_strat == "inter":
                            # Compute embeddings loss on all tracks/detections (track_ids != 0)
                            indices_tuple = self.emb_mining(masked_det_embs, masked_det_targets, masked_track_embs,
                                                            masked_track_targets)
                            sim_loss[h, i] = self.sim_loss(masked_det_embs, masked_det_targets, indices_tuple,
                                                           masked_track_embs,
                                                           masked_track_targets)
                        elif self.contrastive_loss_strat == "valid_inter_intra":
                            # Compute embeddings loss on all valid tracks/detections (track_ids >= 0)
                            valid_tracks = masked_track_targets >= 0
                            valid_dets = masked_det_targets >= 0
                            embeddings = torch.cat([masked_track_embs[valid_tracks], masked_det_embs[valid_dets]],
                                                   dim=0)
                            labels = torch.cat([masked_track_targets[valid_tracks], masked_det_targets[valid_dets]],
                                               dim=0)
                            indices_tuple = self.emb_mining(embeddings, labels)
                            sim_loss[h, i] = self.sim_loss(embeddings, labels, indices_tuple)
                        elif self.contrastive_loss_strat == "valid_inter":
                            # Compute embeddings loss on all valid tracks/detections (track_ids >= 0)
                            valid_tracks = masked_track_targets >= 0
                            valid_dets = masked_det_targets >= 0
                            indices_tuple = self.emb_mining(masked_det_embs[valid_dets], masked_det_targets[valid_dets],
                                                            masked_track_embs[valid_tracks],
                                                            masked_track_targets[valid_tracks])
                            sim_loss[h, i] = self.sim_loss(masked_det_embs[valid_dets], masked_det_targets[valid_dets],
                                                           indices_tuple, masked_track_embs[valid_tracks],
                                                           masked_track_targets[valid_tracks])
                        else:
                            raise NotImplementedError
                    elif self.loss_strat == "infoNCE":
                        # Compute embeddings loss on all tracks/detections (track_ids >= 0)
                        valid_tracks = masked_track_targets >= 0
                        valid_dets = masked_det_targets >= 0
                        embeddings = torch.cat([masked_track_embs[valid_tracks], masked_det_embs[valid_dets]], dim=0)
                        labels = torch.cat([masked_track_targets[valid_tracks], masked_det_targets[valid_dets]], dim=0)
                        sim_loss[h, i] = self.sim_loss(embeddings, labels)
                    else:
                        raise NotImplementedError

                if hasattr(self, "classifier") and len(masked_det_embs) != 0:
                    mask_cls_loss[h, i] = True

                    # Compute cls loss on all detections
                    inputs = dets.confs[i, dets.masks[i]]
                    targets = (masked_det_targets >= 0).to(torch.float32)
                    weights = torch.ones(len(targets), device=self.device)
                    weights[targets == 0] /= 2 * (targets == 0).sum()
                    weights[targets == 1] /= 2 * (targets == 1).sum()
                    cls_loss[h, i] = self.bce(inputs, targets, weight=weights)

        # Compute mean losses over valid items in the batch
        sim_loss = sim_loss[mask_sim_loss].mean()
        cls_loss = cls_loss[mask_cls_loss].mean()

        # Handle NaN values
        sim_loss = sim_loss.nan_to_num(0)
        cls_loss = cls_loss.nan_to_num(0)

        return sim_loss, cls_loss

    def log_loss(self, cls_loss, sim_loss, step):
        loss_dict = {}
        if hasattr(self, "classifier"):
            loss_dict[f"{step}/sim_loss"] = sim_loss
            loss_dict[f"{step}/cls_loss"] = cls_loss
        loss = self.alpha_loss * sim_loss + (1 - self.alpha_loss) * cls_loss
        loss_dict[f"{step}/loss"] = loss
        self.log_dict(
            loss_dict,
            on_epoch=True,
            on_step="train" == step,
            prog_bar="train" == step,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_cfg.init_lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches / 10,  # FIXME very slow, call a lot of get_items
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_callbacks(self):
        callbacks = [SimMetrics()]
        if hasattr(self, "classifier"):
            callbacks.append(ClsMetrics())
        return callbacks

    def on_save_checkpoint(self, checkpoint):
        # Add custom attributes to the checkpoint dictionary
        checkpoint['computed_sim_threshold'] = self.computed_sim_threshold
        checkpoint['computed_sim_threshold2'] = self.computed_sim_threshold2
        checkpoint['best_distr_overlap_threshold'] = self.best_distr_overlap_threshold

    def on_load_checkpoint(self, checkpoint):
        # Load custom attributes from the checkpoint dictionary
        self.computed_sim_threshold = checkpoint.get('computed_sim_threshold', None)
        self.computed_sim_threshold2 = checkpoint.get('computed_sim_threshold2', None)
        self.best_distr_overlap_threshold = checkpoint.get('best_distr_overlap_threshold', None)

    def on_validation_end(self):
        # self.final_tracking_threshold = self.computed_sim_threshold2
        self.final_tracking_threshold = self.computed_sim_threshold
        # self.final_tracking_threshold = self.best_distr_overlap_threshold
        log.info(f"Final tracking threshold set to {self.final_tracking_threshold}")
        return