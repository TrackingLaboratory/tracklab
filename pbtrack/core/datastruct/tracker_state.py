from contextlib import AbstractContextManager
from pathlib import Path
import pickle
import zipfile
import pandas as pd

from .tracking_dataset import TrackingSet
import logging

log = logging.getLogger(__name__)


class TrackerState(AbstractContextManager):
    save_columns = [
        "image_id",
        "id",
        "bbox_ltwh",
        "bbox_c",
        "keypoints_xyc",
        "person_id",
        "category_id",
        "video_id",
        "embeddings",
        "visibility_scores",
        "body_masks",
    ]

    def __init__(self, tracking_set: TrackingSet, filename="trackerstate.pklz"):
        self.gt = tracking_set
        self.predictions = None
        self.filename = Path(filename)
        self.zf = None
        self.video_id = None

    def __call__(self, video_id):
        self.video_id = video_id
        return self

    def __enter__(self):
        self.zf = zipfile.ZipFile(
            self.filename,
            mode="a",
            compression=zipfile.ZIP_STORED,  # Don't compress pickle
            allowZip64=True,
        )
        return super().__enter__()

    def update(self, detections):
        if self.predictions is None:
            self.predictions = detections
        else:
            self.predictions = pd.concat([self.predictions, detections])

    def save(self):
        """
        Saves a pickle in a zip file if the video_id is not yet stored in it.
        """
        assert self.video_id is not None, "Save can only be called in a contextmanager"
        assert (
            self.predictions is not None
        ), "The predictions should not be empty when saving"
        if f"{self.video_id}.pkl" not in self.zf.namelist():
            with self.zf.open(f"{self.video_id}.pkl", "w") as fp:
                predictions = self.predictions[
                    self.predictions.video_id == self.video_id
                ]
                predictions = predictions[self.save_columns]
                pickle.dump(predictions, fp, protocol=pickle.DEFAULT_PROTOCOL)
        else:
            log.info(f"{self.video_id} already exists in pklz file")

    def load(self):
        """
        Returns:
            bool: True if the pickle contains the video detections,
                and False otherwise.
        """
        assert self.video_id is not None, "Load can only be called in a contextmanager"
        if f"{self.video_id}.pkl" in self.zf.namelist():
            with self.zf.open(f"{self.video_id}.pkl", "r") as fp:
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
        self.zf.close()
        self.zf = None
        self.video_id = None
