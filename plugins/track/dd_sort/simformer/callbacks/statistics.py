import pytorch_lightning as pl
import wandb
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import roc_curve

log = logging.getLogger(__name__)


class PairsStatistics(pl.Callback):
    def __init__(self, plot=False):
        super().__init__()
        self.plot = plot
        self.positive_sim = []
        self.negative_sim = []

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        # Determine best threshold
        best_threshold = find_threshold_overlap(self.positive_sim, self.negative_sim)
        plt = plot_distributions_with_threshold(self.positive_sim, self.negative_sim, best_threshold)
        pl_module.best_distr_overlap_threshold = best_threshold
        log.info(f"Best distr_overlap_threshold found on validation set: {best_threshold:.3f}")

        log_dict = {}
        log_dict[f"val/distr_overlap_opt_th"] = best_threshold
        pl_module.log_dict(log_dict, logger=True, on_step=False, on_epoch=True)

        pl_module.logger.experiment.log({"val/sim_distr": wandb.Image(plt)})
        if self.plot:
            plt.show()
        plt.close()

    def on_validation_batch_end(self,
                                trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule",
                                outputs: Optional[STEP_OUTPUT],
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0,):

        td_sim_matrix = outputs['td_sim_matrix']

        gt_assoc_matrices = compute_gt_assoc_matrices(outputs['dets'], outputs['tracks'])

        pos_sim = td_sim_matrix[gt_assoc_matrices.to(bool)]
        pos_sim = pos_sim[pos_sim != -float("inf")]
        neg_sim = td_sim_matrix[~gt_assoc_matrices.to(bool)]
        neg_sim = neg_sim[neg_sim != -float("inf")]
        self.positive_sim.extend(pos_sim.tolist())
        self.negative_sim.extend(neg_sim.tolist())


def find_threshold_overlap(positive_similarities, negative_similarities, num_thresholds=1000):
    min_sim = min(min(positive_similarities), min(negative_similarities))
    max_sim = max(max(positive_similarities), max(negative_similarities))
    thresholds = np.linspace(min_sim, max_sim, num_thresholds)

    # Convert lists to numpy arrays for vectorized operations
    positive_similarities = np.array(positive_similarities)
    negative_similarities = np.array(negative_similarities)

    # Compute the number of false negatives for each threshold
    # False negatives occur when positive similarities are below the threshold.
    fn_matrix = positive_similarities[:, None] < thresholds
    fn_counts = fn_matrix.sum(axis=0) / len(positive_similarities)

    # Compute the number of false positives for each threshold
    # False positives occur when negative similarities are above the threshold.
    fp_matrix = negative_similarities[:, None] > thresholds
    fp_counts = fp_matrix.sum(axis=0) / len(negative_similarities)

    overlap = fn_counts + fp_counts
    best_threshold = thresholds[np.argmin(overlap)]

    return best_threshold


def find_threshold_roc(positive_distances, negative_distances, w=2):
    y_true = [1] * len(positive_distances) + [0] * len(negative_distances)
    y_scores = positive_distances + negative_distances
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    # Use the modified Youden's J statistic
    J = tpr - w * fpr
    optimal_idx = np.argmax(J[1:]) + 1  # Exclude the first threshold value which might be skewed by -inf

    return thresholds[optimal_idx]


def plot_distributions_with_threshold(positive_similarities, negative_similarities, threshold):
    # Weights for positive similarities to represent percentages
    weights_pos = np.ones_like(positive_similarities) / len(positive_similarities)
    # Weights for negative similarities to represent percentages
    weights_neg = np.ones_like(negative_similarities) / len(negative_similarities)

    plt.figure()
    # Plot positive similarities in green
    plt.hist(positive_similarities, weights=weights_pos, color='green', alpha=0.5, bins=30, label='Positive Similarities')
    # Plot negative similarities in red
    plt.hist(negative_similarities, weights=weights_neg, color='red', alpha=0.5, bins=30, label='Negative Similarities')

    plt.axvline(x=threshold, color='blue', linestyle='--')
    # Display threshold value to the right of the threshold line
    plt.text(threshold + 0.02, plt.gca().get_ylim()[1] * 0.95, f'{threshold:.3f}', color='blue', ha='right')

    # Setting axis labels, title, and legend
    plt.xlabel("Similarity")
    plt.ylabel("Percentage of similarities")
    plt.title("Distributions of Positive and Negative Similarities")
    plt.legend(loc='upper left')

    # Show the plot
    plt.tight_layout()
    return plt


def compute_gt_assoc_matrices(tracks, dets):
    # Vectorized computation for the ground truth assignment matrices
    valid_track_identity = tracks.targets.unsqueeze(-1) != -1
    valid_det_identity = dets.targets.unsqueeze(1) != -1
    valid_identity = valid_track_identity & valid_det_identity
    matches = dets.targets.unsqueeze(-1) == dets.targets.unsqueeze(1)
    gt_assoc_matrices = (matches & valid_identity).int()
    return gt_assoc_matrices
