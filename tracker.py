from curses import meta
from dataclasses import dataclass, field
from enum import Enum
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

@dataclass
class Detection:
    metadata_id: Optional[int] = None
    source: Optional[Source] = None
    bbox_id: Optional[int] = None
    keypoint_id: Optional[int] = None
    personid_id: Optional[int] = None
    
    def add_tracker(self, tracker):
        self.tracker = tracker
    
    @property
    def metadata(self):
        assert hasattr(self, "tracker")
        return self.tracker.metadata[self.metadata_id]
    
    @metadata.setter
    def metadata(self, metadata):
        assert hasattr(self, "tracker")
        self.tracker.metadata.append(metadata)
        self.metadata_id = len(self.tracker.metadata)

    @property
    def bbox(self):
        assert hasattr(self, "tracker")
        return self.tracker.bboxes[self.bbox_id]
    
    @property
    def keypoint(self):
        assert hasattr(self, "tracker")
        return self.tracker.keypoints[self.keypoint_id]
    
    @property
    def person_id(self):
        assert hasattr(self, "tracker")
        return self.tracker.reid_features[self.personid_id]


@dataclass
class Tracker:
    detections: list[Detection] = field(default_factory=list)
    metadata: list[Metadata] = field(default_factory=list)
    _dataframe: Optional[pd.DataFrame] = None
    _dirty: bool = True

    @property
    def dataframe(self):
        if True: # self._dirty:
            self._dataframe = pd.DataFrame(self.detections)
            self._dirty = False

        return self._dataframe

    def add_detection(self, detection: Detection):
        detection.add_tracker(self)
        self.detections.append(detection)
        self._dirty = True



def main():
    tracker = Tracker()
    
    for i in range(100_000):
        detection = Detection()
        tracker.add_detection(detection)
        detection.metadata = Metadata(filename=f"file_{i}", height=100, width=100)
        print(detection)
    
    return tracker
    
if __name__ == "__main__":
    print(main().dataframe.head())
