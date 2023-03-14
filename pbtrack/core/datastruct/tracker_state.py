import os
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
from contextlib import AbstractContextManager
from os.path import abspath
from pathlib import Path

from .detections import Detections
from .tracking_dataset import TrackingSet
from pbtrack.utils.coordinates import kp_to_bbox_w_threshold, bbox_ltrb2ltwh


import logging

log = logging.getLogger(__name__)


class TrackerState(AbstractContextManager):
    LOAD_COLUMNS = dict(
        detection=[
            "image_id",
            "id",
            "bbox_ltwh",
            "bbox_c",
            "keypoints_xyc",
            "category_id",
            "video_id",
        ],
        reid=[
            "embeddings",
            "visibility_scores",
            "body_masks",
        ],
        tracking=["track_bbox_tlwh", "track_bbox_conf", "track_id", "person_id"],
    )

    def __init__(
        self,
        tracking_set: TrackingSet,
        load_file=None,
        json_file=None,  # TODO merge with above behavior
        save_file=None,
        load_from_groundtruth=False,
        compression=zipfile.ZIP_STORED,
        load_step=None,
        bbox_format=None,
    ):
        assert load_step in [
            "detect_single",
            "detect_multi",
            "reid",
            None,
        ], "Load_step must be one of 'detect_single', 'detect_multi', 'reid', 'None'"
        self.gt = tracking_set
        self.predictions = None
        # self.filename = Path(filename)
        self.load_file = Path(load_file) if load_file else None
        self.save_file = Path(save_file) if save_file else None
        self.compression = compression
        self.LOAD_COLUMNS["reid"] += self.LOAD_COLUMNS["detection"]
        self.LOAD_COLUMNS["tracking"] += self.LOAD_COLUMNS["reid"]
        if load_step:
            self.load_columns = (
                self.LOAD_COLUMNS[load_step]
                if load_step not in ["detect_multi", "detect_single"]
                else self.LOAD_COLUMNS["detection"]
            )
        self.zf = None
        self.video_id = None
        self.do_detect_multi = load_step is None
        self.do_detect_single = load_step == "detect_multi" or self.do_detect_multi
        self.do_reid = load_step == "detect_single" or self.do_detect_single
        self.do_tracking = load_step == "reid" or self.do_reid
        self.bbox_format = bbox_format

        self.json_file = json_file
        if self.json_file is not None:
            self.load_predictions_from_json(json_file)

        self.load_from_groundtruth = load_from_groundtruth
        if self.load_from_groundtruth:
            self.load_groundtruth(load_step)

    def load_groundtruth(self, load_step):
        # FIXME only work for topdown -> handle bottomup
        # We consider here that detect_multi detects the bbox
        # and that detect_single detects the keypoints
        assert (
            load_step != "reid"
        ), "Cannot load from groundtruth in reid step. Can only load bboxes or keypoints"
        self.gt_detections = self.gt.detections.copy()
        if load_step == "detect_multi":
            self.gt_detections["keypoints_xyc"] = pd.NA
            self.gt_detections["track_id"] = pd.NA
            self.gt_detections.drop(columns=["track_id"], inplace=True)
            self.gt_detections.rename(columns={"visibility": "bbox_c"}, inplace=True)
        elif load_step == "detect_single":
            self.gt_detections["track_id"] = pd.NA
            self.gt_detections.drop(columns=["track_id"], inplace=True)
            self.gt_detections.rename(columns={"visibility": "bbox_c"}, inplace=True)

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
        predictions.bbox_ltwh = predictions.bbox_ltwh.apply(lambda x: np.array(x))
        predictions["id"] = predictions.index
        predictions.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
        predictions.keypoints_xyc = predictions.keypoints_xyc.apply(
            lambda x: np.reshape(np.array(x), (-1, 3))
        )
        if self.bbox_format == "ltrb":
            # TODO tracklets coming from Tracktor++ are in ltbr format
            predictions.loc[
                predictions["bbox_ltwh"].notna(), "bbox_ltwh"
            ] = predictions[predictions["bbox_ltwh"].notna()].bbox_ltwh.apply(
                lambda x: bbox_ltrb2ltwh(x)
            )
        predictions.loc[predictions["bbox_ltwh"].isna(), "bbox_ltwh"] = predictions[
            predictions["bbox_ltwh"].isna()
        ].keypoints_xyc.apply(lambda x: kp_to_bbox_w_threshold(x, vis_threshold=0))
        predictions["bbox_c"] = predictions.keypoints_xyc.apply(
            lambda x: x[:, 2].mean()
        )
        predictions = predictions.merge(
            self.gt.image_metadatas[["video_id"]],
            how="left",
            left_on="image_id",
            right_index=True,
        )
        self.json_predictions = Detections(predictions)
        if self.do_tracking:
            self.json_predictions.drop(
                ["track_id"], axis=1, inplace=True
            )  # TODO NEED TO DROP track_id if we want to perform tracking
        else:
            self.json_predictions["track_bbox_kf_ltwh"] = self.json_predictions[
                "bbox_ltwh"
            ]  # FIXME config to decide if track_bbox_kf_ltwh or bbox_ltwh should be used

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
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
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
            self.predictions = pd.concat(
                [self.predictions, detections]
            )  # TODO UPDATE should update existing rows or append if new rows

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
            return self.json_predictions[
                self.json_predictions.video_id == self.video_id
            ]
        if self.load_from_groundtruth:
            return self.gt_detections[self.gt_detections.video_id == self.video_id]
        if self.load_file is None:
            return None

        log.info(f"loading from {self.load_file}")
        assert self.video_id is not None, "Load can only be called in a contextmanager"
        if f"{self.video_id}.pkl" in self.zf["load"].namelist():
            with self.zf["load"].open(f"{self.video_id}.pkl", "r") as fp:
                video_detections = pickle.load(fp)
                self.update(video_detections)
                return video_detections[self.load_columns]
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
