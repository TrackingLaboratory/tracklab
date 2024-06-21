import logging

from .common import MOT

log = logging.getLogger(__name__)

categories_list = [
    {'id': 1, 'name': 'pedestrian'},
]


class DanceTrack(MOT):
    def __init__(self, dataset_path: str, nvid: int = -1,
                 vids_dict: list = None, public_dets_subpath : str = None, *args, **kwargs):
        log.info(f"Loading DanceTrack dataset from {dataset_path}.")
        super().__init__(dataset_path, categories_list, nvid, vids_dict, public_dets_subpath, *args, **kwargs)
