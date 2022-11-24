from hydra.utils import instantiate

from pbtrack.datasets.loaders.dataset_loader import DatasetLoader
from pbtrack.datasets.loaders.posetrack21_loader import PoseTrack21Loader
from pbtrack.tracker.tracker import Tracker


def load_tracker(dataset_cfg) -> Tracker:
    ds_loader = instantiate(dataset_cfg)
    tracker = ds_loader.build_tracker()
    return tracker
