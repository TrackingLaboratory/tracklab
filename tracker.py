from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass
class Metadata:
    filename: str
    height: int
    width: int
    video_id: str # or int ?
    frame: int
    frames: int

@dataclass
class Bbox:
    xc: float
    yc: float
    w: float
    h: float
    conf: float

@dataclass
class Keypoint:
    x: float
    y: float
    conf: float

class Source(Enum):
    GT = 1
    DET = 2
    TRACK = 3


@dataclass
class Detection:
    metadata: Metadata
    bbox: Bbox
    keypoint: Keypoint
    source: Source
    id_features: np.ndarray # store as numpy or as torch ?


@dataclass
class Tracker:
    detections: list[Detection] = []