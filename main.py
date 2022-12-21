import logging

import hydra
from hydra.utils import instantiate

import torch

from pbtrack.core.datastruct.tracker_state import TrackerState

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def track(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO support Mac chips

    train_reid = False  # FIXME put in config
    train_pose = False  # FIXME put in config

    # Init
    tracking_dataset = instantiate(cfg.dataset)
    model_pose = instantiate(cfg.detection, device=device)
    model_reid = instantiate(
        cfg.reid,
        tracking_dataset=tracking_dataset,
        device=device,
        model_pose=model_pose,
    )
    model_track = instantiate(cfg.track, device=device)
    evaluator = instantiate(cfg.eval)
    vis_engine = instantiate(cfg.visualization)

    # Train reid
    if train_reid:
        model_reid.train()

    # Train pose
    if train_pose:
        model_pose.train()

    tracker_state = TrackerState(tracking_dataset.val_set)

    # Run tracking
    tracking_engine = instantiate(
        cfg.engine,
        detector=model_pose,
        reider=model_reid,
        tracker=model_track,
        tracker_state=tracker_state,
        vis_engine=vis_engine,
    )
    tracking_engine.run()

    # Evaluation
    evaluator.run(tracker_state)


if __name__ == "__main__":
    track()
