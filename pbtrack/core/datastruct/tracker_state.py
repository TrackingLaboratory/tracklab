import json
from contextlib import AbstractContextManager
from os.path import abspath
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pandas as pd

from .tracking_dataset import TrackingSet
from .detections import Detections
import logging

log = logging.getLogger(__name__)


class TrackerState(AbstractContextManager):
    SAVE_COLUMNS = dict(
        detection=[
            "image_id",
            "id",
            "bbox_ltwh",
            "bbox_c",
            "keypoints_xyc",
            "category_id",
            "person_id",
            "video_id",
        ],
        reid=[
            "embeddings",
            "visibility_scores",
            "body_masks",
        ],
        tracking=["track_bbox_tlwh", "track_bbox_conf", "track_id"],
    )

    def __init__(
        self,
        tracking_set: TrackingSet,
        load_file=None,
        json_file=None,  # TODO merge with above behavior
        save_file=None,
        compression=zipfile.ZIP_STORED,
        load_step="reid",
    ):
        self.gt = tracking_set
        self.predictions = None
        # self.filename = Path(filename)
        self.load_file = Path(load_file) if load_file else None
        self.save_file = Path(save_file) if save_file else None
        self.compression = compression
        self.SAVE_COLUMNS["reid"] += self.SAVE_COLUMNS["detection"]
        self.SAVE_COLUMNS["tracking"] += self.SAVE_COLUMNS["reid"]
        self.save_columns = self.SAVE_COLUMNS[load_step]
        self.zf = None
        self.video_id = None
        self.do_detection = load_step == None
        self.do_reid = load_step == "detection" or self.do_detection
        self.do_tracking = load_step == "reid" or self.do_reid

        self.json_file = json_file
        if self.json_file is not None:
            self.load_predictions_from_json(json_file)

    def load_predictions_from_json(self, json_file):
        anns_path = Path(json_file)
        anns_files_list = list(anns_path.glob("*.json"))
        assert len(anns_files_list) > 0, "No annotations files found in {}".format(
            anns_path
        )
        detections = []
        for path in anns_files_list:
            with open(path) as json_file:
                data_dict = json.load(json_file)
                detections.extend(data_dict["annotations"])
        predictions = pd.DataFrame(detections)
        predictions.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
        predictions['track_bbox_kf_ltwh'] = predictions['bbox_ltwh']  # FIXME config to decide if track_bbox_kf_ltwh or bbox_ltwh should be used
        predictions.dropna(subset=['bbox_ltwh'], inplace=True)
        predictions['id'] = predictions.index
        predictions.bbox_ltwh = predictions.bbox_ltwh.apply(lambda x: np.array(x))
        predictions.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
        predictions.keypoints_xyc = predictions.keypoints_xyc.apply(
            lambda x: np.reshape(np.array(x), (-1, 3))
        )
        predictions['bbox_c'] = predictions.keypoints_xyc.apply(lambda x: x[:, 2].mean())
        predictions = predictions.merge(
            self.gt.image_metadatas[["video_id"]], how="left", left_on="image_id", right_index=True
        )
        self.predictions = Detections(predictions)
        self.do_detection = False
        self.do_reid = False
        self.do_tracking = False

    def __call__(self, video_id):
        self.video_id = video_id
        return self

    def __enter__(self):
        self.zf = {}
        if self.load_file is None:
            load_zf = None
        else:
            load_zf = zipfile.ZipFile(
                self.load_file,
                mode="r",
                compression=self.compression,
                allowZip64=True,
            )

        if self.save_file is None:
            save_zf = None
        else:
            save_zf = zipfile.ZipFile(
                self.save_file,
                mode="a",
                compression=self.compression,
                allowZip64=True,
            )

        if (self.load_file is not None) and (self.load_file == self.save_file):
            # Fix possible bugs when loading and saving from same file
            zf = zipfile.ZipFile(
                self.load_file,
                mode="a",
                compression=self.compression,
                allowZip64=True,
            )
            self.zf = dict(load=zf, save=zf)
        else:
            self.zf = dict(load=load_zf, save=save_zf)
        return super().__enter__()

    def update(self, detections):
        if self.predictions is None:
            self.predictions = detections
        else:
            self.predictions = self.predictions[
                ~(self.predictions["video_id"] == self.video_id)
            ]
            self.predictions = pd.concat([self.predictions, detections])

    def save(self):
        """
        Saves a pickle in a zip file if the video_id is not yet stored in it.
        """
        if self.save_file is None:
            return
        log.info(f"saving to {abspath(self.save_file)}")
        assert self.video_id is not None, "Save can only be called in a contextmanager"
        assert (
            self.predictions is not None
        ), "The predictions should not be empty when saving"
        if f"{self.video_id}.pkl" not in self.zf["save"].namelist():
            with self.zf["save"].open(f"{self.video_id}.pkl", "w") as fp:
                predictions = self.predictions[
                    self.predictions.video_id == self.video_id
                ]
                pickle.dump(predictions, fp, protocol=pickle.DEFAULT_PROTOCOL)
        else:
            log.info(f"{self.video_id} already exists in {self.save_file} file")

    def load(self):
        """
        Returns:
            bool: True if the pickle contains the video detections,
                and False otherwise.
        """
        if self.json_file is not None:
            return self.predictions[self.predictions.video_id == self.video_id]
        if self.load_file is None:
            return None

        log.info(f"loading from {self.load_file}")
        assert self.video_id is not None, "Load can only be called in a contextmanager"
        if f"{self.video_id}.pkl" in self.zf["load"].namelist():
            with self.zf["load"].open(f"{self.video_id}.pkl", "r") as fp:
                video_detections = pickle.load(fp)
                self.update(video_detections)
                return video_detections[self.save_columns]
        else:
            log.info(f"{self.video_id} not in pklz file")
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        """
        TODO : remove all heavy data associated to a video_id
        """
        for zf_type in ["load", "save"]:
            if self.zf[zf_type] is not None:
                self.zf[zf_type].close()
                self.zf[zf_type] = None
        self.video_id = None
