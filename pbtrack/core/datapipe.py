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
            image = self.metadatas.loc[detection.image_id]
            return self.model.preprocess(image=image, detection=detection)
        else:
            image = self.metadatas.iloc[idx]
            sample = (image.id, self.model.preprocess(image))
            return sample