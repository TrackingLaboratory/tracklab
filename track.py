import logging
from pathlib import Path
import hydra
from hydra.utils import instantiate
import torch
from pbtrack.tracker.tracker import Tracker
from tqdm import tqdm

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def track(cfg):
    if cfg.save_imgs:
        imgs_name = Path("imgs")

    if cfg.save_vid:
        vid_name = None  # WHY ?

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_pose = instantiate(cfg.detection, device=device)
    model_reid = instantiate(cfg.reid, device=device, model_pose=model_pose)
    model_track = instantiate(cfg.track, device=device)

    dataset = instantiate(cfg.dataset)

    model_reid.train()

    # process images, but don't do it like this :
    all_detections = []

    for data in tqdm(dataset, desc="Inference"):  # tensor RGB (3, H, W)
        detections = model_pose.run(data)
        detections = model_reid.run(detections, data)
        detections = model_track.run(data, detections)
        all_detections.extend(detections)

    tracker = Tracker([det.asdict() for det in all_detections])

    vis_engine = instantiate(cfg.visualization, tracker=tracker)

    for data in tqdm(dataset, desc="Visualization"):
        vis_engine.process(data)


if __name__ == "__main__":
    track()
