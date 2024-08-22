import logging
from collections import OrderedDict
from itertools import islice
from math import ceil

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from torch.utils.data import Sampler

log = logging.getLogger(__name__)


class SimFormerSampler(Sampler):
    def __init__(self, dataset, batch_size=128, num_samples=8, **kwargs):
        super().__init__(dataset)
        self.rng = np.random.default_rng()
        self.dataset = dataset
        assert hasattr(dataset, "samples"), "You should define the samples"
        self.samples = self.dataset.samples
        assert len(np.unique([x["global_track_id"] for x in self.samples])) == len(self.samples), "All tracklets should have different IDs"
        self.track_ids = np.sort([x["global_track_id"] for x in self.samples])
        self.video_ids = np.sort(np.unique([x["video_id"] for x in self.samples]))
        # self.image_ids = np.sort(np.unique([x["image_id"] for x in self.samples]))
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dl_batch_size = batch_size * num_samples

    def __len__(self):
        return ceil(len(self.video_ids) / self.batch_size)

    def sample_generator(self):
        """Generator of tuples with sample_idx and random image_id"""
        random_video_ids = self.rng.choice(self.video_ids, len(self.video_ids), replace=False)
        samples = pd.DataFrame(self.samples)

        for video_id in random_video_ids:
            video_samples = samples[samples["video_id"] == video_id]
            possible_image_ids = np.unique(np.concatenate(np.array(video_samples["image_id"])))
            image_id = self.rng.choice(possible_image_ids)
            for i in range(self.num_samples):
                try:
                    sample_index = video_samples.index[i]
                except IndexError:
                    sample_index = -1
                yield sample_index, image_id

    def __iter__(self):
        yield from batched(self.sample_generator(), self.dl_batch_size)


class ValSampler(Sampler):
    def __init__(self, dataset, batch_size=128):
        super().__init__(dataset)
        self.dataset = dataset
        self.samples = pd.DataFrame(dataset.samples)
        self.image_ids = np.unique(self.samples.image_id)
        self.batch_size = batch_size
        self.num_samples = self.samples.groupby("image_id")["detections"].count().max()
        self.dl_batch_size = batch_size * self.num_samples

    def __len__(self):
        return (len(self.samples) + self.dl_batch_size - 1) // self.dl_batch_size

    def __iter__(self):
        for img_ids in batched(self.image_ids, self.batch_size):
            batched_samples = []
            for img_id in img_ids:
                samples = self.samples[self.samples.image_id == img_id]
                samples = samples.index
                samples = np.pad(samples, pad_width=(0, self.num_samples-len(samples)),
                                 constant_values=-1)
                batched_samples.extend(samples)
            if (len(batched_samples) % self.batch_size) != 0:
                batched_samples = np.pad(batched_samples,
                                         pad_width=(0, self.batch_size - (len(batched_samples) % self.batch_size)),
                                         constant_values=-1)
            yield batched_samples


class HarderSimFormerSampler(SimFormerSampler):
    def __init__(self, ids_bias=10, mix_frames=True, **kwargs):
        super().__init__(**kwargs)
        self.ids_bias = ids_bias
        self.mix_frames = mix_frames

    def __len__(self):
        if not self.mix_frames:
            return (len(self.samples) + self.num_samples - 1) // self.num_samples
        else:
            return super().__len__()
    def __iter__(self):
        samples = pd.DataFrame(self.samples)
        if "id_switch" not in samples.columns:
            raise ValueError("You should recreate the pklz with 'id_switch'")
        p = samples.groupby("image_id").id_switch.sum() * self.ids_bias + 1
        p = p / p.sum()
        hard_percentage = p[samples.groupby("image_id").id_switch.sum() > 0].sum()
        idxs = self.rng.choice(len(self.image_ids), len(self.image_ids) + 10, p=p, replace=True)
        calculated_percentage = (samples.groupby("image_id").id_switch.sum().to_numpy()[idxs]>0).sum() / len(idxs)
        log.info(f"Hard samples percentage : {hard_percentage * 100:.2f}% (theoretical); {calculated_percentage * 100:.2f}% (real)")
        batch = []
        while len(idxs) > 0:
            while len(idxs) > 0 and len(batch) < self.dl_batch_size:
                idx, *idxs = idxs  # Oz <3
                current_image_id = self.image_ids[idx]
                batch_samples = samples[samples.image_id == current_image_id]
                batch.extend(batch_samples.index)
                if not self.mix_frames:
                    if (len(batch) % self.num_samples) != 0:
                        batch = list(np.pad(batch, (0, self.num_samples - (len(batch) % self.num_samples)), constant_values=-1))
            if len(batch) < self.dl_batch_size:
                log.warning(f"Thrown out last batch (ids: {batch})")
                break
            yield batch[:self.dl_batch_size]
            batch = batch[self.dl_batch_size:]


class HardSimFormerSampler(SimFormerSampler):
    def __iter__(self):
        samples = pd.DataFrame(self.samples)
        p = samples.groupby("image_id").detections.count()
        p = p / p.sum()
        idxs = self.rng.choice(len(self.image_ids), len(self.image_ids), p=p, replace=True)
        batch = []
        for i in range(len(self)):
            while len(batch) < self.dl_batch_size:
                idx, *idxs = idxs  # Oz <3
                current_image_id = self.image_ids[idx]
                batch_samples = samples[samples.image_id == current_image_id]
                batch.extend(batch_samples.index)
            yield batch[:self.dl_batch_size]
            batch = batch[self.dl_batch_size:]


class RandomFrameSampler(Sampler):
    def __init__(self, dataset, batch_size=128, num_samples=8):
        super().__init__(dataset)
        self.rng = np.random.default_rng(1234)
        self.dataset = dataset
        assert hasattr(dataset, "samples"), "You should define the samples"
        self.samples = self.dataset.samples
        self.image_ids = np.unique([x["image_id"] for x in self.samples])
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dl_batch_size = batch_size * num_samples

    def __len__(self):
        return (len(self.samples) + self.dl_batch_size - 1) // self.dl_batch_size

    def __iter__(self):
        # use ordered sets and dicts to make this function deterministic when seed is set
        samples = pd.DataFrame(self.samples)
        # group set of image ids by video id
        video_id_to_image_id_set = samples.sort_values(['video_id', 'image_id']).groupby('video_id')['image_id'].apply(OrderedSet).to_dict()
        video_id_to_image_id_set = OrderedDict(sorted(video_id_to_image_id_set.items()))
        # pop image ids from video_id_to_image_id_set until none remaining
        pairs_list_for_epoch = []
        while len(video_id_to_image_id_set) > 0:
            # build one training sample
            # a training sample is a list of "num_samples" pairs of tracklet-detection
            # it correspond to an instance of a tracklet-to-detection association occurring in online tracking
            videos_to_pick_from = OrderedSet(video_id_to_image_id_set.keys())
            chosen_pairs = []
            while len(chosen_pairs) < self.num_samples and len(videos_to_pick_from) > 0:
                # pick a random video
                video_id = self.rng.choice(list(videos_to_pick_from))
                videos_to_pick_from.remove(video_id)
                # pick a random frame from that video
                image_id = self.rng.choice(list(video_id_to_image_id_set[video_id]))
                video_id_to_image_id_set[video_id].remove(image_id)
                if len(video_id_to_image_id_set[video_id]) == 0:
                    video_id_to_image_id_set.pop(video_id)
                # pick all pairs from that frame
                pairs = samples[samples['image_id'] == image_id]
                chosen_pairs.extend(list(pairs.index))

            if len(chosen_pairs) >= self.num_samples:
                chosen_pairs = chosen_pairs[:self.num_samples]  # FIXME not optimal, some pairs are lost
                pairs_list_for_epoch.extend(chosen_pairs)

        yield from batched(pairs_list_for_epoch, self.dl_batch_size)


samplers = {
    "simple": SimFormerSampler,
    "hard": HardSimFormerSampler,
    "harder": HarderSimFormerSampler,
    "val": SimFormerSampler,  # ValSampler,
    "random_frame": RandomFrameSampler,
}


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.

    >>> list(batched('ABCDEFG', 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    On Python 3.12 and above, this is an alias for :func:`itertools.batched`.
    """
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch
