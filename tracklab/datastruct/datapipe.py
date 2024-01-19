from torch.utils.data import Dataset
from tracklab.utils.cv2 import cv2_load_image


class EngineDatapipe(Dataset):
    def __init__(self, model) -> None:
        self.model = model
        self.image_filepaths = None
        self.img_metadatas = None
        self.detections = None

    def update(self, image_filepaths: dict, img_metadatas, detections):
        del self.img_metadatas
        del self.detections
        self.image_filepaths = image_filepaths
        self.img_metadatas = img_metadatas
        self.detections = detections

    def __len__(self):
        if self.model.level == "detection":
            return len(self.detections)
        elif self.model.level == "image":
            return len(self.img_metadatas)
        else:
            raise ValueError(f"You should provide the appropriate level for you module not '{self.model.level}'")

    def __getitem__(self, idx):
        if self.model.level == "detection":
            detection = self.detections.iloc[idx]
            metadata = self.img_metadatas.loc[detection.image_id]
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = (
                detection.name,
                self.model.preprocess(image=image, detection=detection, metadata=metadata),
            )
            return sample
        elif self.model.level == "image":
            metadata = self.img_metadatas.iloc[idx]
            if self.detections is not None and len(self.detections) > 0:
                detections = self.detections[self.detections.image_id == metadata.name]
            else:
                detections = self.detections
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = (self.img_metadatas.index[idx], self.model.preprocess(
                image=image, detections=detections, metadata=metadata))
            return sample
        else:
            raise ValueError("Please provide appropriate level.")
