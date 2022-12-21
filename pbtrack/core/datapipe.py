from torch.utils.data import Dataset


class EngineDatapipe(Dataset):
    def __init__(self, model) -> None:
        self.model = model
        self.img_metadatas = None
        self.detections = None

    def update(self, img_metadatas, detections=None):
        del self.img_metadatas
        del self.detections
        self.img_metadatas = img_metadatas
        self.detections = detections

    def __len__(self):
        if self.detections is not None:
            return len(self.detections)
        else:
            return len(self.img_metadatas)

    def __getitem__(self, idx):
        if self.detections is not None:
            detection = self.detections.iloc[idx]
            metadata = self.img_metadatas.loc[detection.image_id]
            sample = (
                detection.name,
                self.model.preprocess(detection=detection, metadata=metadata),
            )
            return sample
        else:
            metadata = self.img_metadatas.iloc[idx]
            sample = (metadata.id, self.model.preprocess(metadata))
            return sample
