from pbtrack.datastruct import Metadatas, Detections
from typing import Optional
from torch.utils.data import Dataset

class EngineDatapipe(Dataset):
    def __init__(self, model, metadatas: Metadatas, detections:Optional[Detections] = None) -> None:
        self.model = model
        self.metadatas = metadatas
        self.detections = detections

    def __len__(self):
        return len(self.metadatas)

    def __getitem__(self, idx):
        if self.detections is not None:
            detection = self.detections.iloc[idx]
            metadata = self.metadatas.loc[detection.image_id]
            sample = (idx, self.model.preprocess(detection=detection, metadata=metadata))
            return sample
        else:
            metadata = self.metadatas.iloc[idx]
            sample = (metadata.id, self.model.preprocess(metadata))
            return sample