from typing import List, Optional

import numpy as np
import pandas as pd
from overrides import overrides
from torch import nn


class Transform:
    def __init__(self):
        self.rng: Optional[np.random.Generator] = None

    def __call__(self, df: pd.DataFrame, video_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented the __call__ function."
        )

    def set_rng(self, rng=None):
        if self.rng is not None:
            return
        self.rng = rng or np.random.default_rng()
        if hasattr(self, "transforms"):
            for transform in self.transforms:
                transform.set_rng(self.rng)


class BatchTransform(Transform, nn.Module):
    @overrides
    def __call__(self, batch):
        raise NotImplementedError(f"{type(self).__name__} has not implemented the __call__ function.")


class OfflineTransforms:
    transform_functions = {}

    @classmethod
    def register(cls, name, function):
        cls.transform_functions[name] = function

    @classmethod
    def get_transforms(cls, names):
        return [cls.transform_functions[name] for name in names]


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        self.set_rng()
        a = args[0]
        for transform in self.transforms:
            a = transform(*args, **kwargs)

        return a


class NoOp(Transform):
    def __call__(self, *args, **kwargs):
        return args[0]


class SomeOf(Transform):
    def __init__(self, transforms: List[Transform],
                 min_choice: int = 1,
                 max_choice: Optional[int] = None):
        super().__init__()
        self.transforms = transforms
        self.min_choice = min_choice
        self.max_choice = max_choice or len(self.transforms)

    def __call__(self, *args, **kwargs):
        self.set_rng()
        a = args[0]
        size_choice = self.rng.integers(self.min_choice, self.max_choice)
        transforms = self.rng.choice(self.transforms, size=size_choice)
        for transform in transforms:
            a = transform(*args, **kwargs)

        return a

class ProbabilisticTransform(Transform):
    def __init__(self, transforms: List[Transform], probs: Optional[List[float]] = None):
        super().__init__()
        self.transforms = transforms
        if probs is not None:
            assert len(transforms) == len(probs), "Probs must have the same length as transforms."
            self.probs = probs
        else:
            self.probs = [0.] * len(transforms)

    def __call__(self, *args, **kwargs):
        self.set_rng()
        a = args[0]
        for transform, prob in zip(self.transforms, self.probs):
            if self.rng.random() < prob:
                a = transform(*args, **kwargs)
        return a
