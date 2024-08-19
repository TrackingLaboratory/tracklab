from dd_sort.simformer.transforms import BatchTransform
import torch


class AppMixup(BatchTransform):
    def __init__(self, std=0.005):
        super().__init__()
        self.std = std

    def __call__(self, batch):
        embs = batch["track_feats"]["embeddings"]
        other = torch.randperm(embs.shape[0])
        other_embeddings = embs[other]
        other_embeddings[other_embeddings.isnan()] = 0
        alpha = torch.abs(
            torch.normal(mean=0, std=self.std, size=(len(embs),), device=embs.device)
        )
        batch["track_feats"]["embeddings"] = (1 - alpha)[..., None, None, None] * embs + alpha[
            ..., None, None, None] * other_embeddings
        return batch


class AppAddNoise(BatchTransform):
    def __init__(self, std=0.005):
        super().__init__()
        self.std = std

    def __call__(self, batch):
        embs = batch["track_feats"]["embeddings"]
        batch["track_feats"]["embeddings"] = embs + torch.normal(mean=0, std=self.std, size=embs.shape,
                                                                 device=embs.device)
        return batch
