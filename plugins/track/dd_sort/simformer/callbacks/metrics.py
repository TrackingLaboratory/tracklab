import pytorch_lightning as pl
import torch
import torchmetrics
import logging
import wandb
from torch import Tensor
from torchmetrics import Metric

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class SimMetrics(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.acc = Accuracy().to(pl_module.device)
        self.roc = torchmetrics.classification.BinaryROC()
        self.auroc = torchmetrics.classification.BinaryAUROC()
        
        # fixme new roc check if it is ok and clean
        self.running_sim_matrix = torch.tensor([], device=pl_module.device)
        self.running_gt_matrix = torch.tensor([], dtype=torch.bool, device=pl_module.device)
        self.running_tracks_mask = torch.tensor([], dtype=torch.bool, device=pl_module.device)
        self.running_dets_mask = torch.tensor([], dtype=torch.bool, device=pl_module.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        tracks = outputs["tracks"]
        dets = outputs["dets"]
        td_sim_matrix = outputs["td_sim_matrix"]
        gt_ass_matrix = tracks.targets.unsqueeze(2) == dets.targets.unsqueeze(1)

        # roc
        valid_idx = tracks.masks.unsqueeze(2) * dets.masks.unsqueeze(1)
        valid_sim_matrix = outputs["td_sim_matrix"][valid_idx]
        valid_gt_ass_matrix = gt_ass_matrix[valid_idx].to(torch.int32)
        self.roc.update(valid_sim_matrix, valid_gt_ass_matrix)
        self.auroc.update(valid_sim_matrix, valid_gt_ass_matrix)

        intersection = torch.eq(tracks.targets.unsqueeze(dim=2), dets.targets.unsqueeze(dim=1))
        inter_tracks_masks = intersection.any(dim=2)
        inter_dets_masks = intersection.any(dim=1)

        # acc
        binary_ass_matrix, _ = pl_module.association(td_sim_matrix, inter_tracks_masks, inter_dets_masks)  # FIXME use best th from previous epoch

        idx = torch.arange(td_sim_matrix.numel(), device=td_sim_matrix.device).reshape(td_sim_matrix.shape)
        preds = idx[binary_ass_matrix]
        targets = idx[gt_ass_matrix]
        self.acc.update(preds, targets)

        # fixme new roc check if it is ok and clean
        self.running_sim_matrix = torch.cat((self.running_sim_matrix, td_sim_matrix), dim=0)
        self.running_gt_matrix = torch.cat((self.running_gt_matrix, gt_ass_matrix), dim=0)
        self.running_tracks_mask = torch.cat((self.running_tracks_mask, tracks.masks), dim=0)
        self.running_dets_mask = torch.cat((self.running_dets_mask, dets.masks), dim=0)

    def on_validation_epoch_end(self, trainer, pl_module):
        best_roc_threshold = log_roc(self.roc, self.auroc, pl_module, trainer.current_epoch, "sim")
        pl_module.computed_sim_threshold = best_roc_threshold
        log.info(f"Best computed_sim_threshold found on validation set: {best_roc_threshold:.3f}")
        pl_module.log_dict({"val/best_roc_threshold": best_roc_threshold}, logger=True, on_step=False,
                           on_epoch=True)
        pl_module.log_dict(
            {"val/sim_acc": self.acc.compute().item()}, logger=True, on_step=False, on_epoch=True
        )

        # fixme new roc check if it is ok and clean
        thresholds = torch.linspace(0., 1., 101, device=pl_module.device)
        accuracies = torch.zeros_like(thresholds, device=pl_module.device)
        for i, th in enumerate(thresholds):
            binary_ass_matrix, _ = pl_module.association(
                self.running_sim_matrix, self.running_tracks_mask, self.running_dets_mask, th
            )
            correct = binary_ass_matrix == self.running_gt_matrix
            correct_tracks = torch.all(correct, dim=1)[self.running_tracks_mask]
            correct_dets = torch.all(correct, dim=2)[self.running_dets_mask]
            accuracies[i] = (correct_tracks.sum() + correct_dets.sum()) / (self.running_tracks_mask.sum() + self.running_dets_mask.sum())
        computed_sim_threshold2 = thresholds[torch.argwhere(accuracies == torch.amax(accuracies)).flatten()[-1]]
        pl_module.computed_sim_threshold2 = computed_sim_threshold2
        pl_module.log_dict({"val/computed_sim_threshold2": computed_sim_threshold2}, logger=True, on_step=False, on_epoch=True)
        log.info(f"Best computed_sim_threshold2 found on validation set: {pl_module.computed_sim_threshold2:.3f}")

class ClsMetrics(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.roc = torchmetrics.classification.BinaryROC()
        self.auroc = torchmetrics.classification.BinaryAUROC()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        dets = outputs["dets"]
        preds = dets.confs[dets.masks]
        targets = (dets.targets[dets.masks] >= 0).to(torch.int32)
        self.roc.update(preds, targets)
        self.auroc.update(preds, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        best_roc_cls_threshold = log_roc(self.roc, self.auroc, pl_module, trainer.current_epoch, "cls")
        pl_module.best_roc_cls_threshold = best_roc_cls_threshold
        log.info(f"Best best_roc_cls_threshold found on validation set: {best_roc_cls_threshold:.3f}")

class Accuracy(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


def log_roc(roc, auroc, pl_module, epoch, name):
    fpr, tpr, thresholds = roc.compute()
    best_threshold = thresholds[torch.argmax(tpr - fpr)]
    b_auroc = auroc.compute()
    fig_, ax_ = roc.plot(score=True)
    ax_.set_aspect("equal")
    ax_.set_title(f"val/{name}_ROC - epoch {epoch}")
    idx = [i for i in range(0, len(fpr), max(1, len(fpr) // 20))]
    for i, j in enumerate(idx):
        ax_.annotate(
            f"{thresholds[j]:.3f}",
            xy=(fpr[j], tpr[j]),
            xytext=((-1) ** i * 20, (-1) ** (i + 1) * 20),
            textcoords="offset points",
            fontsize=6,
            arrowprops=dict(arrowstyle="->"),
            ha="center",
        )
    log_dict = {}
    log_dict[f"val/{name}_opt_th"] = best_threshold
    log_dict[f"val/{name}_auroc"] = b_auroc.item()
    pl_module.log_dict(log_dict, logger=True, on_step=False, on_epoch=True)
    pl_module.logger.experiment.log({f"val/{name}_ROC": wandb.Image(fig_)})
    plt.close(fig_)
    return best_threshold
