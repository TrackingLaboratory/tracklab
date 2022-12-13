from pbtrack.datastruct import ImageMetadatas, Detections
from typing import Optional
from torch.utils.data import Dataset


class EngineDatapipe(Dataset):
    def __init__(
        self, model, metadatas: ImageMetadatas, detections: Optional[Detections] = None
    ) -> None:
        self.model = model
        self.metadatas = metadatas
        self.detections = detections

    def __len__(self):
        if self.detections is not None:
            return len(self.detections)
        else:
            return len(self.metadatas)

    def __getitem__(self, idx):
        if self.detections is not None:
            detection = self.detections.iloc[idx]
            metadata = self.metadatas.loc[detection.image_id]
            sample = (
                detection.name,
                self.model.preprocess(detection=detection, metadata=metadata),
            )
            return sample
        else:
            metadata = self.metadatas.iloc[idx]
            sample = (metadata.id, self.model.preprocess(metadata))
            return sample
