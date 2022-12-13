import hydra
import logging

import pandas as pd
import torch
from hydra.utils import instantiate
from pbtrack.datastruct.tracker_state import TrackerState
from pbtrack.core import EngineDatapipe
from pbtrack.core.tracking_engine import OnlineTrackingEngine
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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
    # evaluator = instantiate(cfg.eval)
    # vis_engine = instantiate(cfg.visualization)
    trainer = pl.Trainer()

    # Train reid
    if train_reid:
        model_reid.train()

    # Train pose
    if train_pose:
        model_pose.train()

    tracker_state = TrackerState(tracking_dataset.val_set)
    # Run tracking
    imgs_meta = tracker_state.gt.image_metadatas
    for video_id in tracker_state.gt.video_metadatas.id:
        detection_datapipe = DataLoader(
            dataset=EngineDatapipe(
                model_pose, imgs_meta[imgs_meta.video_id == video_id]
            ),
            batch_size=2,
        )
        model_track.reset()
        tracking_engine = OnlineTrackingEngine(
            model_pose, model_reid, model_track, imgs_meta
        )
        detections_list = trainer.predict(
            tracking_engine, dataloaders=detection_datapipe
        )
        detections = pd.concat(detections_list)
        tracker_state.update(detections)

    # Evaluation
    # evaluator.run(tracking_dataset)

    # Visualization
    # vis_engine.run(tracking_dataset)


if __name__ == "__main__":
    track()
