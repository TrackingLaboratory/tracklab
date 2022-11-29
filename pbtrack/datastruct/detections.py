import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional
from pbtrack.utils.coordinates import kp_img_to_kp_bbox


#### TODO remove/comment below
class Source(Enum):
    METADATA = 0  # just an image
    DET = 1  # model detection
    TRACK = 2  # model tracking


@dataclass
class Metadata:
    file_path: str  # path/video_name/file.jpg
    height: int
    width: int
    image_id: Optional[str] = None  # for eval of posetrack
    file_name: Optional[str] = None  # file.jpg
    video_name: Optional[str] = None  # video_name
    frame: Optional[int] = None
    nframes: Optional[int] = None


@dataclass
class Bbox:
    x: float
    y: float
    w: float
    h: float
    conf: float


@dataclass
class Keypoint:
    x: float
    y: float
    conf: float
    part: int


@dataclass
class Detection:
    metadata: Optional[Metadata] = None
    source: Optional[Source] = 0
    bbox: Optional[Bbox] = None
    keypoints: Optional[list] = None
    reid_features: Optional[np.ndarray] = None
    visibility_score: Optional[np.ndarray] = None
    body_mask: Optional[np.ndarray] = None
    person_id: Optional[int] = -1

    def asdict(self):
        keypoints = {}
        if self.keypoints:
            for i, keypoint in enumerate(self.keypoints):
                keypoints = {**keypoints,
                             **{f"kp{i}_{k}": v for k, v in asdict(keypoint).items()},
                             }
        bboxes = {}
        if self.bbox:
            bboxes = {f"bb_{k}": v for k, v in asdict(self.bbox).items()}
        return {
            'source': self.source,
            'person_id': self.person_id,
            **asdict(self.metadata),
            **bboxes,
            **keypoints,
        }

    # TODO change location of those functions
    def rescale_xy(self, coords, input_shape, output_shape=None):
        if output_shape is None:
            output_shape = (self.metadata.height, self.metadata.width)
        x_ratio = output_shape[1] / input_shape[1]
        y_ratio = output_shape[0] / input_shape[0]
        coords[:, 0] *= x_ratio
        coords[:, 1] *= y_ratio
        return coords

    def bbox_xyxy(self, image_shape=None):
        xyxy = [self.bbox.x, self.bbox.y, self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h]
        if image_shape is not None:
            x_ratio = image_shape[1] / self.metadata.width
            y_ratio = image_shape[0] / self.metadata.height
            xyxy[[0, 2]] *= x_ratio
            xyxy[[1, 3]] *= y_ratio
        return np.array(xyxy)

    def bbox_xywh(self, image_shape=None):
        xywh = [self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h]
        if image_shape is not None:
            x_ratio = image_shape[1] / self.metadata.width
            y_ratio = image_shape[0] / self.metadata.height
            xywh[[0, 2]] *= x_ratio
            xywh[[1, 3]] *= y_ratio
        return np.array(xywh)
#### TODO remove/comment above

class Detections(pd.DataFrame):
    rcols = {
        'id': True,
        'person_id': True,
        'category_id': True,
        'image_id': True,
        'video_id': True,
        'bbox_ltwh': False,
        'bbox_ltrb': False,
        'bbox_cxcywh': False,
        'bbox_head': False,
        'keypoints_xyc': False,
        'keypoints_bbox_xyc': False,
        'visibility': False,
    }

    def __init__(self, *args, **kwargs):
        super(Detections, self).__init__(*args, **kwargs)

    # Required for Dataframe subclassing
    @property
    def _constructor(self):
        return Detections

    @property
    def _constructor_sliced(self):
        return DetectionsSeries

    @property
    def aaa_base_class_view(self):
        # use this to view the base class, needed for debugging in some IDEs.
        return pd.DataFrame(self)

    # Utils for converting between formats
    @property
    def bbox_ltrb(self):
        """Converts from (left, top, width, heights) to (left, top, right, bottom)"""
        return self.bbox_ltwh.apply(
            lambda ltwh: np.concatenate((ltwh[:2], ltwh[:2] + ltwh[2:]))
        )

    @property
    def bbox_cmwh(self):
        """Converts from (left, top, width, heights) to (horizontal center, vertical middle, width, height)"""
        return self.bbox_ltwh.apply(
            lambda ltwh: np.concatenate((ltwh[:2] + ltwh[2:] / 2, ltwh[2:]))
        )

    @property
    def keypoints_bbox_xyc(self):
        """Converts from keypoints in image coordinates to keypoints in bbox coordinates"""
        return self.bbox_ltwh.apply(
            lambda r: kp_img_to_kp_bbox(r.keypoints_xyc, r.bbox_ltwh), axis=1
        )


class DetectionsSeries(pd.Series):
    @property
    def _constructor(self):
        return DetectionsSeries

    @property
    def _constructor_expanddim(self):
        return Detections
