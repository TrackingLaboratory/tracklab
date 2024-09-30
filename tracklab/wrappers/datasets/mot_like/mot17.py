import logging

from .common import MOT

log = logging.getLogger(__name__)

categories_list = [
    {'id': 1, 'name': 'pedestrian'},
    {'id': 2, 'name': 'person_on_vehicle'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'bicycle'},
    {'id': 5, 'name': 'motorbike'},
    {'id': 6, 'name': 'non_mot_vehicle'},
    {'id': 7, 'name': 'static_person'},
    {'id': 8, 'name': 'distractor'},
    {'id': 9, 'name': 'occluder'},
    {'id': 10, 'name': 'occluder_on_ground'},
    {'id': 11, 'name': 'occluder_full'},
    {'id': 12, 'name': 'reflection'},
]


class MOT17(MOT):
    name = "MOT17"
    nickname = "m17"

    def __init__(self, dataset_path: str, nvid: int = -1, nframes: int = -1,
                 vids_dict: list = None, public_dets_subpath : str = None, *args, **kwargs):
        log.info(f"Loading MOT17 dataset from {dataset_path}.")
        super().__init__(dataset_path, categories_list, nvid, nframes, vids_dict, public_dets_subpath, *args, **kwargs)
