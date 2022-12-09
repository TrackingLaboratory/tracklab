import hydra
import logging
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

    # Init
    tracking_dataset = instantiate(cfg.dataset)
    model_pose = instantiate(cfg.detection, device=device)
    model_reid = instantiate(cfg.reid, tracking_dataset=tracking_dataset, device=device, model_pose=model_pose)
    model_track = instantiate(cfg.track, device=device) 
    
    tracking_engine = OnlineTrackingEngine(model_pose, model_reid, model_track, tracking_dataset.val_set.metadatas)
    detection_datapipe = DataLoader(dataset=EngineDatapipe(model_pose, tracking_dataset.val_set.metadatas),
                                    batch_size=8
    )
    trainer = pl.Trainer()
    trainer.predict(tracking_engine, dataloaders=detection_datapipe)

    # evaluator = instantiate(cfg.eval) # TODO
    # vis_engine = instantiate(cfg.visualization) # TODO

    # Train Reid # TODO
    #if train_reid:
    #    model_reid.train()

    # Train Pose # TODO
    #if train_pose:
    #    model_pose.train()

    # Tracking engine # TODO
    #tracking_state = TrackerState(tracking_dataset.val_set)
    #model_pose.run(tracking_state)  # list Image or slice Images OR batch numpy images -> list Detection or slice Detections
    #model_reid.run(tracking_state)  # list Detection or slice Detections OR batch numpy images + skeletons + masks -> list Detection or slice Detections
    #model_track.run(tracking_state)  # online: list frame Detection -> list frame Detection | offline: video detections -> video detections

    # Performance # TODO
    # evaluator.run(tracker)

    # Visualization # TODO
    # vis_engine.run(tracker)


if __name__ == "__main__":
    track()