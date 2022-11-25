import hydra
import logging
import torch
from hydra.utils import instantiate

from pbtrack.tracker.tracker import Tracker

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def track(cfg, train_reid=True, train_pose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO support Mac chips

    # Init
    tracking_dataset = instantiate(cfg.dataset)
    model_pose = instantiate(cfg.detection, device=device)
    model_reid = instantiate(cfg.reid, tracking_dataset=tracking_dataset, device=device, model_pose=model_pose)
    model_track = instantiate(cfg.track, device=device)
    # evaluator = instantiate(cfg.eval)
    # vis_engine = instantiate(cfg.visualization)

    # Train Reid
    if train_reid:
        model_reid.train()

    # Train Pose
    if train_pose:
        model_pose.train()

    # Tracking
    tracker = Tracker(tracking_dataset.val_set)
    model_pose.run(tracker)
    model_reid.run(tracker)
    model_track.run(tracker)

    # Performance
    # evaluator.run(tracker)

    # Visualization
    # vis_engine.run(tracker)


if __name__ == "__main__":
    track()
