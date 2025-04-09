import logging

from .common import MOT

log = logging.getLogger(__name__)

categories_list = [
    {'id': 1, 'name': 'pedestrian'},
]


class DanceTrack(MOT):
    # 40 train videos
    # 25 val videos
    # 35 test videos
    """
    Public detections notes:
    - train:
        - my_det.txt: official YoloX weights from DanceTrack, trained on train set, ran by Baptiste to get these detections
    - val:
        - my_det.txt: official YoloX weights from DanceTrack, trained on train set, ran by Baptiste to get these detections
        - yolox_dets.txt: official YoloX detections from GHOST, trained on train set
    - test
        - my_det.txt: official YoloX weights from DanceTrack, trained on train set, ran by Baptiste to get these detections
        - yolox_dets.txt: official YoloX detections from GHOST, trained on train (+val?) set
    """
    name = "DanceTrack"
    nickname = "dt"

    def __init__(self, dataset_path: str, nvid: int = -1, nframes: int = -1,
                 vids_dict: list = None, public_dets_subpath : str = None, *args, **kwargs):
        log.info(f"Loading DanceTrack dataset from {dataset_path}.")
        super().__init__(dataset_path, categories_list, nvid, nframes, vids_dict, public_dets_subpath, *args, **kwargs)
