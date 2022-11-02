from dataclasses import asdict, dataclass, field
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
    part: int

@dataclass
class Detection:
    filename: Optional[str] = None
    source: Optional[Source] = None
    bbox_id: Optional[int] = None
    keypoint_id: Optional[int] = None
    keypoint_part: Optional[int] = None
    personid_id: Optional[int] = None
    
    def add_tracker(self, tracker: "Tracker"):
        self.tracker = tracker
    
    def asdict(self):
        return {
            "source": self.source,
            **asdict(self.metadata),
            **{f"bb_{k}":v for k,v in asdict(self.bbox).items()},
            **{f"kp_{k}":v for k,v in asdict(self.keypoint).items()},}
            # person_id=self.person_id, Too much information
    
    @property
    def metadata(self):
        assert hasattr(self, "tracker")
        return self.tracker.metadata[self.filename]
    
    @metadata.setter
    def metadata(self, metadata: Metadata):
        assert hasattr(self, "tracker")
        self.filename = metadata.filename
        self.tracker.metadata[self.filename] = metadata
        self.tracker._dirty = True

    @property
    def bbox(self):
        assert hasattr(self, "tracker")
        return self.tracker.bboxes[self.bbox_id]

    @bbox.setter
    def bbox(self, bbox: Bbox):
        assert hasattr(self, "tracker")
        self.bbox_id = id(bbox)
        self.tracker.bboxes[self.bbox_id] = bbox
        self.tracker._dirty = True
    
    # @property
    # def keypoints(self):
    #     return self.tracker.keypoints[self.keypoint_id-self.keypoint_part:self.keypoint_id-self.keypoint_part+17]

    @property
    def keypoint(self):
        assert hasattr(self, "tracker")
        return self.tracker.keypoints[self.keypoint_id]
    

    @keypoint.setter
    def keypoint(self, keypoint: Keypoint):
        assert hasattr(self, "tracker")
        self.keypoint_id = id(keypoint)
        self.keypoint_part = keypoint.part
        self.tracker.keypoints[self.keypoint_id] = keypoint
        self.tracker._dirty = True
    
    @property
    def person_id(self):
        assert hasattr(self, "tracker")
        return self.tracker.reid_features[self.personid_id]

    @person_id.setter
    def person_id(self, reid_feature):
        assert hasattr(self, "tracker")
        self.personid_id = id(reid_feature)
        self.tracker.reid_features[self.personid_id] = reid_feature
        self.tracker._dirty = True

@dataclass
class Tracker:
    detections: list[Detection] = field(default_factory=list)
    metadata: list[Metadata] = field(default_factory=dict)
    bboxes: list[Bbox] = field(default_factory=dict)
    keypoints: list[Keypoint] = field(default_factory=dict)
    reid_features: list[np.ndarray] = field(default_factory=dict)

    _dataframe: Optional[pd.DataFrame] = None
    _dirty: bool = True

    @property
    def dataframe(self):
        if self._dirty:
            self._dataframe = pd.DataFrame([det.asdict() for det in self.detections])
            self._dirty = False

        return self._dataframe
    
    def add_detection(self, detection: Detection):
        detection.add_tracker(self)
        self.detections.append(detection)
        self._dirty = True

    def filter_detections(self, df):
        return np.array(self.detections)[df.index]


def main():
    """Example usage : """

    # create tracker
    tracker = Tracker()
    
    # For each image
    for i in range(1000):
        
        # 1. Run Detector
        
        metadata = Metadata(filename=f"file_{i}", height=100, width=100)
        bbox = Bbox(0,0,0,0,0)

        image_detections: list[Detection] = []
        for part in range(17):
            detection = Detection()
            tracker.add_detection(detection)
            detection.metadata = metadata
            detection.bbox = bbox
            detection.keypoint = Keypoint(x=0,y=0,conf=0, part=part)
            image_detections.append(detection)

        # 2. Run Reid
        reid_array = np.zeros((2048,))
        for detection in image_detections:
            detection.person_id = reid_array


    # print(tracker.detections[99].keypoints)
    df = tracker.dataframe
    print(df[df["filename"]=="file_5"].index)
    print(tracker.filter_detections(df[df["filename"]=="file_5"]))
    
    return tracker
    
if __name__ == "__main__":
    print(main().dataframe.head(100))
