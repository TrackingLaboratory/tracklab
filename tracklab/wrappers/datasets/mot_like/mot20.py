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
    {'id': 13, 'name': 'crowd'}
]


class MOT20(MOT):
    def __init__(self, dataset_path: str, nvid: int = -1,
                 vids_dict: list = None, *args, **kwargs):
        log.info(f"Loading MOT20 dataset from {dataset_path}.")
        super().__init__(dataset_path, categories_list, nvid, vids_dict, *args, **kwargs)
