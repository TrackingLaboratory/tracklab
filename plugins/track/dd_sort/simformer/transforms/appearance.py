from simformer.transforms import BatchTransform
import torch


class AppMixup(BatchTransform):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def __call__(self, batch):
        embeddings = batch["track_feats"]["embeddings"]
        other = torch.randperm(embeddings.shape[0])
        other_embeddings = embeddings[other]
        other_embeddings[other_embeddings.isnan()] = 0
        alpha = torch.abs(
            torch.normal(mean=0, std=self.std, size=(len(embeddings),), device=embeddings.device)
        )
        batch["track_feats"]["embeddings"] = (1 - alpha)[
            ..., None, None, None
        ] * embeddings + alpha[..., None, None, None] * other_embeddings
        return batch
