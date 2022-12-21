import logging

import hydra
from hydra.utils import instantiate

import torch

from pbtrack.core.datastruct.tracker_state import TrackerState
from pbtrack.core import EngineDatapipe
from pbtrack.core.tracking_engine import OnlineTrackingEngine
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy(
    "file_system"
)  # FIXME : why are we using too much file descriptors ?

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
        model_detect=model_pose,
        model_reid=model_reid,
        model_track=model_track,
        tracker_state=tracker_state,
        vis_engine=vis_engine,
    )
    tracking_engine.run()

    # Evaluation
    evaluator.run(tracker_state)


if __name__ == "__main__":
    track()
