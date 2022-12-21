import hydra
from hydra.utils import instantiate

from pbtrack.core.datastruct.tracker_state import TrackerState

import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy(
    "file_system"
)  # FIXME : why are we using too much file descriptors ?

import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO support Mac chips

    # Initiate all the instances
    tracking_dataset = instantiate(cfg.dataset)
    model_detect = instantiate(cfg.detect, device=device)
    model_reid = instantiate(
        cfg.reid,
        tracking_dataset=tracking_dataset,
        device=device,
        model_detect=model_detect,
    )
    model_track = instantiate(cfg.track, device=device)
    vis_engine = instantiate(cfg.visualization)
    evaluator = instantiate(cfg.eval)

    if cfg.train_detect:
        model_detect.train()

    if cfg.train_reid:
        model_reid.train()

    tracker_state = TrackerState(tracking_dataset.val_set)

    # Run tracking and visualization
    tracking_engine = instantiate(
        cfg.engine,
        model_detect=model_detect,
        model_reid=model_reid,
        model_track=model_track,
        tracker_state=tracker_state,
        vis_engine=vis_engine,
    )
    tracking_engine.run()

    # Evaluation
    evaluator.run(tracker_state)


if __name__ == "__main__":
    main()
