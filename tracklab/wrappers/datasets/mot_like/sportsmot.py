import logging

from .common import MOT

log = logging.getLogger(__name__)

categories_list = [
    {'id': 1, 'name': 'pedestrian'},
]


class SportsMOT(MOT):
    # 45 train videos
    # 45 val videos
    # 150 test videos
    """
    Public detections notes:
    - test
        - dets.txt: official detections from deep_eiou
        - diffmot_yolox_x.txt: official detections from diffmot, trained on train set
        - diffmot_yolox_x_mix.txt: official detections from diffmot, trained on train + val set
    - val
        - deep_eiou.txt: official YoloX weights from deep_eiou, ran by Baptiste to get hese detections. Train on train+val set
        - diffmot_yolox_x.txt: official detections from diffmot, trained on train set
        - diffmot_yolox_x_mix.txt: official detections from diffmot, trained on train + val set
    - train
        - deep_eiou.txt: official YoloX weights from deep_eiou, ran by Baptiste to get hese detections. Train on train+val set
    """
    name = "SportsMOT"
    nickname = "sm"

    def __init__(self, dataset_path: str, nvid: int = -1, nframes: int = -1,
                 vids_dict: list = None, public_dets_subpath : str = None, *args, **kwargs):
        log.info(f"Loading SportsMOT dataset from {dataset_path}.")
        super().__init__(dataset_path, categories_list, nvid, nframes, vids_dict, public_dets_subpath, *args, **kwargs)
