import os.path as osp
import os
#import torch
#from torch.utils.data import Dataset

from .pt_sequence import PTSequence


class PTWrapper():
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, mot_dir, dataset_path, vis_threshold=0.0):
        'seq_name, mot_dir, dataset_path, vis_threshold=0.0'

        sequences = os.listdir(mot_dir)

        self._data = []
        for s in sequences:
            self._data.append(PTSequence(s, mot_dir, dataset_path, vis_threshold))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
