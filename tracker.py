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
    source: Optional[Source] = None
    bbox: Optional[Bbox] = None
    keypoints: Optional[list[Keypoint]] = None
    reid_features : Optional[np.ndarray] = None
    person_id: Optional[int] = None

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
            **keypoints,
            }
        
    def rescale_xy(self, coords, input_shape, output_shape=None):
        if output_shape is None:
            output_shape = (self.metadata.height, self.metadata.width)
        x_ratio = output_shape[1]/input_shape[1]
        y_ratio = output_shape[0]/input_shape[0]
        coords[:, 0] *= x_ratio
        coords[:, 1] *= y_ratio
        return coords
    
    def bbox_xyxy(self, image_shape=None):
        xyxy = [self.bbox.x, self.bbox.y, self.bbox.x+self.bbox.w, self.bbox.y+self.bbox.h]
        if image_shape is not None:
            x_ratio = image_shape[1]/self.metadata.width
            y_ratio = image_shape[0]/self.metadata.height
            xyxy[[0,2]] *= x_ratio
            xyxy[[1,3]] *= y_ratio
        return np.array(xyxy)
    
    def bbox_xywh(self, image_shape=None):
        xywh = [self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h]
        if image_shape is not None:
            x_ratio = image_shape[1]/self.metadata.width
            y_ratio = image_shape[0]/self.metadata.height
            xywh[[0,2]] *= x_ratio
            xywh[[1,3]] *= y_ratio
        return np.array(xywh)

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
