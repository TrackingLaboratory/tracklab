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
class Detection:
    # Metadata
    filename: str = ""
    height: int = 0
    width: int = 0
    video_id: str = "" # or int ?
    frame: int = 0
    frames: int = 0
    source: Optional[Source] = None

    # Bbox
    bb_xc: float = np.nan
    bb_yc: float = np.nan
    bb_w: float = np.nan
    bb_h: float = np.nan
    bb_conf: float = np.nan

    # Keypoint
    keypoint : Optional[np.ndarray] = None

    # ReID
    id_features: Optional[np.ndarray] = None


@dataclass
class Tracker:
    detections: list[Detection] = field(default_factory=list)
    _dataframe: Optional[pd.DataFrame] = None
    _dirty: bool = True

    @property
    def dataframe(self):
        if True: # self._dirty:
            self._dataframe = pd.DataFrame(self.detections)
            self._dirty = False

        return self._dataframe

    def add_detection(self, detection):
        self.detections.append(detection)
        self._dirty = True



def main():
    tracker = Tracker()
    
    for _ in range(100_000):
        tracker.add_detection(Detection())
    
    return tracker
    
if __name__ == "__main__":
    print(main().dataframe.head())
