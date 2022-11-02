from dataclasses import asdict, dataclass, field
from enum import Enum
from optparse import Option
from typing import Optional
import numpy as np
import pandas as pd

class Source(Enum):
    GT = 1
    DET = 2
    TRACK = 3

@dataclass
class Metadata:
    filename: str
    height: int
    width: int
    video_id: Optional[str] = None
    frame: Optional[int] = None
    frames: Optional[int] = None

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
    part: int

@dataclass
class Detection:
    metadata: Optional[Metadata] = None
    source: Optional[Source] = None
    bbox: Optional[Bbox] = None
    keypoints: Optional[list[Keypoint]] = None
    reid_features : Optional[np.ndarray] = None

    def asdict(self):
        keypoints = {}
        for i, keypoint in enumerate(self.keypoints):
            keypoints = {**keypoints,
            **{f"kp{i}_{k}":v for k,v in asdict(keypoint).items()},
            }
        return {
            "source": self.source,
            **asdict(self.metadata),
            **{f"bb_{k}":v for k,v in asdict(self.bbox).items()},
            **keypoints,}


class Tracker:
    def __init__(self, detections: list[Detection]):
        self.detections = pd.DataFrame([det.asdict() for det in detections])


def main():
    """Example usage : """

    # create tracker
    detections = []
    # For each image
    for i in range(1000):
        
        # 1. Run Detector
        
        metadata = Metadata(filename=f"file_{i}", height=100, width=100)
        bbox = Bbox(0,0,0,0,0)

        keypoints: list[Keypoint] = []
        for part in range(17):
            keypoints.append(Keypoint(x=0,y=0,conf=0, part=part))
        detection = Detection(metadata=metadata, bbox=bbox, keypoints=keypoints)
        
        # 2. Run Reid
        reid_array = np.zeros((2048,))
        for detection in detections:
            detection.person_id = reid_array
        
        detections.append(detection)
    
    tracker = Tracker(detections=detections)

    df = tracker.detections
    print(df.head(10))
    print(df[df["filename"]=="file_5"].index)
    
    return tracker
    
if __name__ == "__main__":
    print(main().detections.head(100))
