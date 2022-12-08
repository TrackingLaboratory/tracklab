from pbtrack.datastruct import Images, Detections
from typing import Optional
from torch.utils.data import Dataset

class EngineDatapipe(Dataset):
    def __init__(self, model, images:Images, detections:Optional[Detections] = None) -> None:
        self.model = model
        self.images = images
        self.detections = detections

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.detections is not None:
            detection = self.detections.iloc[idx]
            image = self.images.iloc[detection.image_id]
            return self.model.pre_process(image=image, detection=detection)
        else:
            image = self.images.iloc[idx]
            sample = self.model.pre_process(image)
            print(type(sample))
            return sample