import os
import json
import pickle
import zipfile
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas
import pandas as pd
from contextlib import AbstractContextManager
from os.path import abspath
from pathlib import Path

from tracklab.datastruct.tracking_dataset import TrackingSet
from tracklab.utils.coordinates import generate_bbox_from_keypoints, ltrb_to_ltwh

import logging

log = logging.getLogger(__name__)


class TrackerState(AbstractContextManager):
    def __init__(
            self,
            tracking_set: TrackingSet,
            load_file=None,
            json_file=None,  # TODO merge with above behavior
            save_file=None,
            load_from_groundtruth=False,
            load_from_public_dets=False,
            compression=zipfile.ZIP_DEFLATED,
            bbox_format=None,
            pipeline=None,
    ):
        self.pipeline = pipeline or {}
        self.video_metadatas = tracking_set.video_metadatas
        self.image_metadatas = tracking_set.image_metadatas
        self.image_gt = tracking_set.image_gt
        self.image_pred = None
        if tracking_set.detections_gt is None or tracking_set.detections_gt.empty:
            self.detections_gt = pd.DataFrame(columns=["image_id"])
        else:
            self.detections_gt = tracking_set.detections_gt
        self.detections_pred = None
        if hasattr(tracking_set, "detections_public"):
            self.detections_public = tracking_set.detections_public

        self.load_file = Path(load_file) if load_file else None
        self.save_file = Path(save_file) if save_file else None
        if self.save_file is not None:
            log.info(f"Saving TrackerState to {abspath(self.save_file)}")
        self.compression = compression
        load_columns = defaultdict(set)
        if self.load_file:
            with zipfile.ZipFile(self.load_file) as zf:
                if "summary.json" in zf.namelist():
                    with zf.open("summary.json", force_zip64=True) as fp:
                        summary = json.load(fp)
                        if isinstance(summary["columns"], list):
                            load_columns["detection"] = set(summary["columns"])
                        else:
                            load_columns = {k:set(v) for k, v in summary["columns"].items()}
                else:
                    image_file = next((f for f in zf.namelist() if "image" in f), None)
                    detection_file = next(f for f in zf.namelist() if "image" not in f)
                    with zf.open(detection_file, force_zip64=True) as fp:
                        dets = pandas.read_pickle(fp)
                        load_columns["detection"] = set(dets.columns)
                    if image_file is not None:
                        with zf.open(image_file, force_zip64=True) as fp:
                            images = pandas.read_pickle(fp)
                            load_columns["image"] = set(images.columns)
                    else:
                        load_columns["image"] = set()
        elif load_from_groundtruth:
            load_columns["image"] = set(self.image_gt.columns)
            load_columns["detection"] = set(self.detections_gt.columns)
        elif load_from_public_dets:
            load_columns["image"] = set(self.image_gt.columns)
            load_columns["detection"] = set(self.detections_gt.columns)

        self.input_columns = defaultdict(set)
        self.output_columns = defaultdict(set)
        self.forget_columns = defaultdict(list)
        for module in self.pipeline:
            for level in ["image", "detection"]:
                self.input_columns[level] |= (set(module.get_input_columns(level)) - self.output_columns[level])
                self.output_columns[level] |= set(module.get_output_columns(level))
                self.forget_columns[level] += getattr(module, "forget_columns", [])

        self.load_columns = {"detection": list(), "image": list()}
        if self.load_file or load_from_groundtruth or load_from_public_dets:
            self.load_columns["detection"] = list(
                (load_columns["detection"] - self.output_columns["detection"])
                | self.input_columns["detection"]
                | {"image_id", "video_id"})
            self.load_columns["image"] = list(
                (load_columns["image"] - self.output_columns["image"])
                | self.input_columns["image"]
                | {"video_id", "file_path", "frame"})
            log.info(f"Loading {self.load_columns} from {self.load_file}")

        pipeline.validate(self.load_columns)

        self.zf = None
        self.video_id = None
        self.bbox_format = bbox_format

        self.json_file = json_file
        if self.json_file is not None:
            self.load_detections_pred_from_json(json_file)

        self.load_from_groundtruth = load_from_groundtruth
        if self.load_from_groundtruth:
            self.load_groundtruth(self.load_columns)

        self.load_from_public_dets = load_from_public_dets
        if self.load_from_public_dets:
            self.load_public_dets(self.load_columns)


    def load_groundtruth(self, load_columns):
        from tracklab.engine.engine import merge_dataframes
        if self.pipeline.is_empty():
            self.detections_pred_gt = self.detections_gt.copy()  # load all columns if pipeline is empty
            self.image_pred_gt = merge_dataframes(self.image_metadatas.copy(), self.image_gt.copy())
        else:
            if isinstance(self.load_from_groundtruth, Mapping):
                if "detection" in self.load_from_groundtruth:
                    raise ValueError("You can't yet load some detections from the detections")
                load_columns = {k: list(set(self.load_from_groundtruth.get(k, [])) & set(v))
                                for k, v in load_columns.items()
                                }
            if len(load_columns["detection"]) == 0:
                self.detections_pred_gt = pd.DataFrame(columns=["video_id","image_id"])
            else:
                self.detections_pred_gt = self.detections_gt.copy()[
                    self.detections_gt.columns.intersection(load_columns["detection"]+["image_id", "video_id"])
                ]
            self.image_pred_gt = merge_dataframes(
                self.image_metadatas.copy(), self.image_gt.copy()
            )[list(set(load_columns["image"]) | {"video_id", "file_path", "frame"})]
        self.detections_pred_gt = self.detections_pred_gt.reset_index(drop=True)
        self.detections_pred_gt['id'] = self.detections_pred_gt.index


    def load_public_dets(self, load_columns):
        self.detections_pred_public = self.detections_public.copy()
        self.image_pred_public = self.image_metadatas.copy()


    def load_detections_pred_from_json(self, json_file):
        anns_path = Path(json_file)
        anns_files_list = list(anns_path.glob("*.json"))
        assert len(anns_files_list) > 0, "No annotations files found in {}".format(
            anns_path
        )
        detections_pred = []
        for path in anns_files_list:
            with open(path) as json_file:
                data_dict = json.load(json_file)
                detections_pred.extend(data_dict["annotations"])
        detections_pred = pd.DataFrame(detections_pred)
        detections_pred.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
        detections_pred.bbox_ltwh = detections_pred.bbox_ltwh.apply(
            lambda x: np.array(x)
        )
        detections_pred["id"] = detections_pred.index
        detections_pred.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
        detections_pred.keypoints_xyc = detections_pred.keypoints_xyc.apply(
            lambda x: np.reshape(np.array(x), (-1, 3))
        )
        if self.bbox_format == "ltrb":
            # TODO tracklets coming from Tracktor++ are in ltbr format
            detections_pred.loc[
                detections_pred["bbox_ltwh"].notna(), "bbox_ltwh"
            ] = detections_pred[detections_pred["bbox_ltwh"].notna()].bbox_ltwh.apply(
                lambda x: ltrb_to_ltwh(x)
            )
        detections_pred.loc[
            detections_pred["bbox_ltwh"].isna(), "bbox_ltwh"
        ] = detections_pred[detections_pred["bbox_ltwh"].isna()].keypoints_xyc.apply(
            lambda x: generate_bbox_from_keypoints(x, [0.0, 0.0, 0.0])
        )
        detections_pred["bbox_conf"] = detections_pred.keypoints_xyc.apply(
            lambda x: x[:, 2].mean()
        )
        if detections_pred["bbox_conf"].sum() == 0:
            detections_pred["bbox_conf"] = detections_pred.scores.apply(
                lambda x: x.mean()
            )
            # FIXME confidence score in detections_pred.keypoints_xyc is always 0
        detections_pred = detections_pred.merge(
            self.image_metadatas[["video_id"]],
            how="left",
            left_on="image_id",
            right_index=True,
        )
        self.json_detections_pred = pd.DataFrame(detections_pred)
        if self.do_tracking:
            self.json_detections_pred.drop(
                ["track_id"], axis=1, inplace=True
            )  # TODO NEED TO DROP track_id if we want to perform tracking
        else:
            self.json_detections_pred["track_bbox_kf_ltwh"] = self.json_detections_pred[
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
            Path(self.save_file).parent.mkdir(parents= True, exist_ok=True)
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

    def on_video_loop_end(
            self,
            engine: "TrackingEngine",
            video_metadata: pd.Series,
            video_idx: int,
            detections: pd.DataFrame,
            image_pred: pd.DataFrame,
    ):
        self.update(detections, image_pred)
        self.save()

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        log.info("Tracking ended, final TrackerState stats:")
        self.display_stats()

    def update(self, detections: pd.DataFrame, image_metadata):
        if self.detections_pred is None:
            self.detections_pred = detections
            self.image_pred = image_metadata
        else:
            self.detections_pred = self.detections_pred[
                ~(self.detections_pred["video_id"] == self.video_id)
            ]
            self.detections_pred = pd.concat(
                [self.detections_pred, detections]
            )  # TODO UPDATE should update existing rows or append if new rows
            # updating image metadata
            self.image_pred = self.image_pred[
                ~(self.image_pred["video_id"] == self.video_id)
            ]
            self.image_pred = pd.concat(
                [self.image_pred, image_metadata]
            )

    def save(self):
        """
        Saves a pickle in a zip file if the video_id is not yet stored in it.
        """
        if self.save_file is None:
            return
        assert self.video_id is not None, "Save can only be called in a contextmanager"
        assert (
                self.detections_pred is not None
        ), "The detections_pred should not be empty when saving"
        if "body_masks" in self.detections_pred:
            self.detections_pred = self.detections_pred.drop(['body_masks'], axis=1)
        if f"{self.video_id}.pkl" not in self.zf["save"].namelist():
            if "summary.json" not in self.zf["save"].namelist():
                with self.zf["save"].open("summary.json", "w", force_zip64=True) as fp:
                    summary = {"columns": {
                        "detection": list(self.detections_pred.columns),
                        "image": list(self.image_pred.columns),
                        }
                    }
                    summary_bytes = json.dumps(summary, ensure_ascii=False, indent=4).encode(
                        'utf-8')
                    fp.write(summary_bytes)
            if not self.detections_pred.empty:
                with self.zf["save"].open(f"{self.video_id}.pkl", "w", force_zip64=True) as fp:
                    detections_pred = self.detections_pred[
                        self.detections_pred.video_id == self.video_id
                        ]
                    pickle.dump(detections_pred, fp, protocol=pickle.DEFAULT_PROTOCOL)
            if not self.image_pred.empty:
                with self.zf["save"].open(f"{self.video_id}_image.pkl", "w", force_zip64=True) as fp:
                    image_pred = self.image_pred[
                        self.image_pred.video_id == self.video_id
                    ]
                    pickle.dump(image_pred, fp, protocol=pickle.DEFAULT_PROTOCOL)
        else:
            log.info(f"{self.video_id} already exists in {self.save_file} file")

    def load(self):
        """
        Returns:
            bool: True if the pickle contains the video detections,
                and False otherwise.
        """
        from tracklab.engine.engine import merge_dataframes
        assert self.video_id is not None, "Load can only be called in a contextmanager"
        if self.json_file is not None:
            return self.json_detections_pred[
                self.json_detections_pred.video_id == self.video_id
                ]
        video_detections = pd.DataFrame()
        video_image_preds = self.image_metadatas[self.image_metadatas.video_id == self.video_id]
        if self.load_from_groundtruth:
            video_detections = self.detections_pred_gt[self.detections_pred_gt.video_id == self.video_id]
            video_image_preds = self.image_pred_gt[self.image_pred_gt.video_id == self.video_id]
        if self.load_from_public_dets:
            video_detections = self.detections_public[self.detections_pred_public.video_id == self.video_id]
            video_image_preds = self.image_pred_public[self.image_pred_public.video_id == self.video_id]
        if self.load_file is not None:
            if f"{self.video_id}.pkl" in self.zf["load"].namelist():
                with self.zf["load"].open(f"{self.video_id}.pkl", "r", force_zip64=True) as fp:
                    video_detections = pandas.read_pickle(fp)[self.load_columns["detection"]]  # TODO see with Victor if this ok
                    video_detections = video_detections[video_detections['image_id'].isin(video_image_preds.index)]  # load only detections from the required frames (nframes)
            else:  # TODO throw error?
                log.info(f"{self.video_id} detections not in pklz file.")
                video_detections = pd.DataFrame(columns=self.load_columns["detection"])
            if f"{self.video_id}_image.pkl" in self.zf["load"].namelist():
                with self.zf["load"].open(f"{self.video_id}_image.pkl", "r", force_zip64=True) as fp_image:
                    video_images = merge_dataframes(pandas.read_pickle(fp_image), video_image_preds)[self.load_columns["image"]]
                    video_image_preds = video_images[video_images.index.isin(video_image_preds.index)]  # load only images from the required frames (nframes)
            else:
                video_image_preds = self.image_metadatas[self.image_metadatas.video_id == self.video_id]
        self.update(video_detections, video_image_preds)


        return video_detections, video_image_preds

    def __exit__(self, exc_type, exc_value, traceback):
        """
        TODO : remove all heavy data associated to a video_id
        """
        for zf_type in ["load", "save"]:
            if self.zf[zf_type] is not None:
                self.zf[zf_type].close()
                self.zf[zf_type] = None
        self.video_id = None

        if self.detections_pred is not None:
            self.detections_pred = self.detections_pred.drop(
                columns=self.forget_columns,
                errors="ignore"
            )

    def display_stats(self):
        log.info(f"Total # detections: {len(self.detections_pred)} (GT={len(self.detections_gt)})")
        tracked_text = ["Total # detections with track_id: "]
        unique_text = ["Total # track_ids: "]
        if "track_id" in self.detections_pred.columns:
                tracked_text.append(f"{len(self.detections_pred.dropna(subset=['track_id']))}")
                unique_text.append(f"{len(self.detections_pred.track_id.unique())}")
        if "track_id" in self.detections_gt.columns and "person_id" in self.detections_gt.columns:
            tracked_text.append(f" (GT={len(self.detections_gt.dropna(subset=['track_id']))})")
            unique_text.append(f" (GT={len(self.detections_gt.person_id.unique())})")
        log.info("".join(tracked_text))
        log.info("".join(unique_text))