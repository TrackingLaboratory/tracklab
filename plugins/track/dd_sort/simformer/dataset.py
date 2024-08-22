import collections.abc
import json
import logging
import multiprocessing
import pickle
import zipfile
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Any, Callable, List, Optional, NamedTuple, Dict
from filelock import FileLock

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

from dd_sort.simformer.sampler import samplers
from dd_sort.simformer.transforms import Transform, NoOp, OfflineTransforms
from tracklab.datastruct import TrackingDataset, TrackingSet, TrackerState
from tracklab.pipeline import Pipeline

log = logging.getLogger(__name__)

from functools import lru_cache

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


class SimFormerDataset(Dataset):
    feature_columns = [
        ("bbox_ltwh", 4),
        ("bbox_conf", 1),
        ("keypoints_xyc", (17, 3)),
        ("visibility_scores", 6),
        ("age", 1),
        ("embeddings", 3072),
    ]

    def __init__(
        self,
        gallery_path,
        config_file,
        transforms: Optional[Transform] = None,
    ):
        self.gallery_path = Path(gallery_path)
        self.config_file = Path(config_file)
        self.transforms = transforms or NoOp()
        self._zf = None
        log.debug(f"gallery_path {self.gallery_path} cf {self.config_file}")
        with self.config_file.open() as fp:
            self.samples = json.load(fp)

        self.count_ids = 0.0

    def __len__(self):
        return len(self.samples)

    @property
    def zf(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.gallery_path, mode="r")
        return self._zf
    
    @lru_cache(maxsize=1)
    def _load_pickle(self, sample_video_id):
        with self.zf.open(sample_video_id, "r") as fp:
            df = pickle.load(fp)
        return df

    def __getitem__(self, idx):
        idx, image_id = idx
        if idx == -1:
            return self.create_empty_input()
        else:
            sample = self.samples[idx]
        df = self._load_pickle(sample["video_id"])

        df = df.loc[sample["detections"]]
        df = df[df["image_id"] <= image_id]
        assert df.image_id.is_monotonic_increasing, "Tracklets should be in chronological order"
        df["to_match"] = 0
        if len(df) > 0:
            df["to_match"].iloc[-1] = 1
        df = self.transforms(df)
        tracks = df.loc[df.to_match == 0]
        dets = df.loc[df.to_match == 1]
        assert df.image_id.is_monotonic_increasing, "Tracklets should be in chronological order after transform"
        det_features, det_targets = self.features_targets(dets, image_id)
        track_features, track_targets = self.features_targets(tracks, image_id)
        video_id = sample["video_id"].split("_")[-1].split(".")[0]
        video_id = np.array(int(video_id)).reshape(1)
        image_id = np.array(image_id).reshape(1)
        return {
            "track_feats": track_features,
            "track_targets": track_targets,
            "det_feats": det_features,
            "det_targets": det_targets,
            "video_id": video_id,
            "image_id": image_id,
        }

    def features_targets(self, df, current_image_id):
        df["age"] = current_image_id - df["image_id"]
        features = {}
        for feature_name, feature_dim in self.feature_columns:
            if len(df) > 0:
                feature = np.stack(df[feature_name]).astype(np.float32)
                if feature_name == "keypoints_xyc":  # TODO nicer solution?
                    features[feature_name] = feature
                else:
                    features[feature_name] = feature.reshape(len(df), -1)
            else:
                dim = (0, feature_dim) if feature_name != "keypoints_xyc" else (0, *feature_dim)
                features[feature_name] = np.empty(dim, dtype=np.float32)
        features["index"] = df.index.to_numpy()
        targets = np.array(df["person_id"].to_numpy().astype(np.float32))
        if (df.person_id.unique() < 0).all():
            self.count_ids -= 1.
            targets[:] = self.count_ids
            self.count_ids %= -1e5
        return features, targets

    def create_empty_input(self):
        targets = np.empty((0,), dtype=np.float32)

        return {
            "track_feats": empty_features(self.feature_columns),
            "track_targets": targets,
            "det_feats": empty_features(self.feature_columns),
            "det_targets": targets.copy(),
            "video_id": np.empty((0,), dtype=int),
            "image_id": np.empty((0,), dtype=int),
        }


def empty_features(feature_columns):
    features = {}
    for feature_name, feature_dim in feature_columns:
        dim = (0, feature_dim) if feature_name != "keypoints_xyc" else (0, *feature_dim)
        features[feature_name] = np.empty(dim, dtype=np.float32)
    features["index"] = np.empty((0,), dtype=np.int64)
    return features


def pad_dict(batch):
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        new_batch = batch
        for key in elem:
            batch_part = pad_dict([d[key] for d in batch])
            for i, _ in enumerate(batch_part):
                new_batch[i][key] = batch_part[i]
        return new_batch
    else:
        max_len = max(len(s) for s in batch)
        new_batch = []
        for i, sample in enumerate(batch):
            length = len(sample)
            if length == max_len:
                new_batch.append(sample)
            else:
                pad_values = ((0, max_len - length),) + ((0, 0),) * (sample.ndim - 1)
                constant_values = -1 if sample.dtype == int else np.nan
                padded_sample = np.pad(sample, pad_width=pad_values, constant_values=constant_values)
                new_batch.append(padded_sample)
        return new_batch


def collate_fn(batch, batch_size=np.nan):
    """Turn pandas dataframe into dict of variable size"""
    pad_dict(batch)
    batch = default_collate(batch)
    for k, d in batch.items():
        if isinstance(d, collections.abc.Mapping):
            for kk, dd in d.items():
                num_samples = dd.shape[0] // batch_size
                batch[k][kk] = dd.reshape(batch_size, num_samples, *dd.shape[1:])
        else:
            num_samples = d.shape[0] // batch_size
            batch[k] = d.reshape(batch_size, num_samples, *d.shape[1:])
    return batch


class SimFormerDataModule(pl.LightningDataModule):
    """DataModule to create and manage tracking-specific datasets.

    Args:
        detections: ...
        metadatas: ...
        dataset: "train", "val" or "test"
        path: where to save the data to
        samples_per_video: number of samples that will be created per video
        std_age: standard deviation of distribution of the ages of a generated tracklet
        max_length: maximum length of a generated tracklet
        transforms: list of functions that will transform the given tracklets and det.
        reider: model that will create reid embeddings

    FIXME : reider should be created inside a transform
    """

    def __init__(
        self,
        tracking_dataset: TrackingDataset,
        dataset_splits: List[str],
        path: str,
        name: str,
        num_videos: Optional[int] = None,
        samples_per_video: int = 10,
        std_age: float = 3,
        max_length: int = 50,
        transforms: List[Callable] = ("add_detections", "normalize2image"),
        online_transforms: Optional[Dict[str, Transform]] = None,
        pipeline: Pipeline = None,
        tracker_states: Dict[str, Path] = None,
        batch_size: int = 128,
        num_samples: int = 8,
        num_workers: int = 8,
        sampler: str = "simple",  # "simple", "hard", "harder"
        sampler_args: dict = None,
    ):
        super().__init__()
        self.cfg_detections = dict(
            transforms=[t for t in transforms],
            pipeline=str(pipeline),
        )
        if num_videos is None:
            self.cfg = dict(spv=samples_per_video, age=std_age, ml=max_length)
        else:
            self.cfg = dict(
                spv=samples_per_video, age=std_age, ml=max_length, num_videos=num_videos
            )
        self.num_videos = num_videos
        self.samples_per_video = samples_per_video
        self.std_age = std_age
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.sampler = partial(samplers[sampler], **(sampler_args or {}))
        self.val_sampler = samplers["val"]

        self.path = Path(path)
        self.name = name
        self.metadatas = {}
        self.detections = {}
        self.dataset_splits = dataset_splits
        self.detections_paths = {}
        self.dataset_configs = {}
        self.config_locks = {}
        self.tracking_sets = {}
        cfg_name = "_".join([f"{k}{v}" for k, v in self.cfg.items()])
        for ds_split in dataset_splits:
            tracking_set: TrackingSet = tracking_dataset.sets[ds_split]
            if tracking_set is None:
                raise ValueError(f"This dataset doesn't have a {ds_split}.")
            self.tracking_sets[ds_split] = tracking_set
            self.metadatas[ds_split] = tracking_set.image_metadatas
            self.detections[ds_split] = tracking_set.detections_gt
            self.detections_paths[ds_split] = self.path / f"{self.name}_{ds_split}.pklz"
            self.dataset_configs[ds_split] = self.path / f"{self.name}_{ds_split}_tracklets.json"
            self.config_locks[ds_split] = FileLock(
                self.path / f"{self.name}_{ds_split}_tracklets.json.lock"
            )
        self.transforms = OfflineTransforms.get_transforms(transforms) if transforms else []
        self.online_transforms = online_transforms
        self.datasets = {}
        self.pipeline = pipeline
        self.tracker_states = tracker_states

    def prepare_data(self) -> None:
        """Generate data on one single node if it doesn't exist yet.

        You shouldn't add things to self in this function as it might be called
        in only one process (for example during multi-gpu training)
        """
        for ds_split in self.dataset_splits:
            log.info(f"Loading detections from {self.detections_paths[ds_split]}")
            if not self.detections_paths[ds_split].exists():
                self.generate_detections(self.detections, self.metadatas, ds_split)

            # Check if the detections have been generated with the same parameters as before
            with zipfile.ZipFile(self.detections_paths[ds_split], mode="r") as zf:
                with zf.open("configuration.json") as fp:
                    cfg = json.load(fp)
                    if cfg != self.cfg_detections:
                        from deepdiff import DeepDiff

                        # FIXME: this is not working
                        # raise ValueError(
                        #     "An existing dataset "
                        #     "with different parameters exists "
                        #     f"{DeepDiff(cfg, self.cfg)}"
                        # )
                        log.warning(f"An existing dataset with different parameters exists {DeepDiff(cfg, self.cfg)}")

            if not self.dataset_configs[ds_split].exists():
                with self.config_locks[ds_split]:
                    self.generate_dataset(ds_split)

    def setup(self, stage: str) -> None:
        for dataset_split in self.dataset_splits:
            kwargs = {}
            if self.online_transforms is not None and dataset_split in self.online_transforms:
                kwargs = dict(transforms=self.online_transforms[dataset_split])
            self.datasets[dataset_split] = SimFormerDataset(
                self.detections_paths[dataset_split], self.dataset_configs[dataset_split], **kwargs
            )

    def train_dataloader(self):
        sampler = self.sampler(dataset=self.datasets["train"], batch_size=self.batch_size, num_samples=self.num_samples)
        return DataLoader(
            self.datasets["train"],
            num_workers=self.num_workers,
            collate_fn=partial(collate_fn, batch_size=self.batch_size),
            batch_sampler=sampler,
            worker_init_fn=set_worker_sharing_strategy,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            collate_fn=partial(collate_fn, batch_size=self.batch_size),
            num_workers=self.num_workers,
            batch_sampler=self.val_sampler(self.datasets["val"], self.batch_size),
            worker_init_fn=set_worker_sharing_strategy,
            pin_memory=True,
        )

    def generate_detections(self, detections, metadatas, ds_split):
        detections = detections[ds_split]
        metadatas = metadatas[ds_split]
        detections_path = self.detections_paths[ds_split]
        save_zf = zipfile.ZipFile(
            detections_path,
            mode="x",
            allowZip64=True,
            compression=zipfile.ZIP_DEFLATED,
        )
        with save_zf.open("configuration.json", "w") as fp:
            cfg_bytes = json.dumps(self.cfg_detections).encode("utf-8")
            fp.write(cfg_bytes)
        save_zf.close()

        for video_id, video_detections in tqdm(
            detections.groupby("video_id"),
            desc=f"Generate detections for {ds_split} set",
            unit="video",
        ):
            save_zf = zipfile.ZipFile(
                detections_path,
                mode="a",
                allowZip64=True,
            )
            tracker_state = TrackerState(
                self.tracking_sets[ds_split], pipeline=Pipeline([]), load_file=self.tracker_states[ds_split]
            )
            with tracker_state(video_id) as ts:
                detections = video_detections
                for transform in self.transforms:
                    video_detections, video_image_preds = ts.load()
                    detections = transform(detections, metadatas, preds=video_detections, tracker_state=ts, pipeline=self.pipeline)
            with save_zf.open(f"sample_{video_id}.pkl", "w") as fp:
                pickle.dump(detections, fp, protocol=pickle.DEFAULT_PROTOCOL)

            save_zf.close()

    def generate_dataset(self, ds_split):
        metadatas = self.metadatas[ds_split]
        detections_path = self.detections_paths[ds_split]
        detections_zf = zipfile.ZipFile(detections_path, mode="a")
        params = []
        video_list = [vid for vid in detections_zf.namelist() if vid.endswith(".pkl")]
        video_list = islice(video_list, self.num_videos) if self.num_videos else video_list
        for video_id in video_list:
            if video_id.endswith(".pkl"):
                params.append(
                    CreateTrackletParams(
                        detections_path=detections_path,
                        video_id=video_id,
                        metadatas=metadatas,
                    )
                )
        with multiprocessing.Pool(max(self.num_workers, 1)) as pool:
            samples = list(
                chain.from_iterable(
                    tqdm(
                        pool.imap(create_tracklets_from_video, params, chunksize=1),
                        desc=f"Building track association {ds_split} set",
                        unit="video",
                        total=len(params),
                    )
                )
            )

        with self.dataset_configs[ds_split].open("w") as fp:
            json.dump(samples, fp)

    # def __len__(self):
    #     return len(self.df_samples)
    #
    # def __getitem__(self, idx):
    #     df_sample = self.df_samples.loc[self.df_samples.sample_id == idx]
    #     df_sample = self.transforms
    #     for transform in self.transforms:
    #         df_sample = transform(df_sample, self.metadatas)
    #     return df_sample

def create_tracklets_from_video(params):
    detections_path, video_id, metadatas = params
    video_id_num = int(video_id.split("_")[-1].split(".")[0])  # remove string stuff
    samples = []
    with zipfile.ZipFile(detections_path, mode="r") as detections_zf:
        with detections_zf.open(video_id) as pkl:
            video_detections = pickle.load(pkl)

        for track_id, track_detections in video_detections.groupby("track_id"):
            if track_id >= 0:
                samples.append(
                    {
                        "video_id": str(video_id),
                        "track_id": int(track_id),
                        "global_track_id": int(video_id_num*1000) + int(track_id),
                        "image_id": [int(x) for x in track_detections["image_id"]],
                        "detections": [x for x in track_detections.index],
                    }
                )

    return samples

class CreateTrackletParams(NamedTuple):
    detections_path: Path
    video_id: str
    metadatas: Any

class CreateSamplesParams(NamedTuple):
    detections_path: Path
    video_id: str
    samples_per_video: int
    std_age: float
    max_length: int
    metadatas: Any


def create_samples_from_video(params):
    detections_path, video_id, samples_per_video, std_age, max_length, metadatas = params
    samples = []
    with zipfile.ZipFile(detections_path, mode="r") as detections_zf:
        rng = np.random.default_rng()
        with detections_zf.open(video_id) as pkl:
            video_detections = pickle.load(pkl)

        image_ids = sorted(video_detections.image_id.unique())
        samples_per_video = len(
            image_ids[2:]) if samples_per_video == -1 else samples_per_video
        hard_chosen_image_ids = []
        if "id_switch" in video_detections.columns:
            switches_per_image = video_detections.groupby("image_id").id_switch.sum()[2:]
            if switches_per_image.sum() > 0:
                prob_per_image = switches_per_image.astype(np.float64) / switches_per_image.astype(np.float64).sum()
                hard_chosen_image_ids = sorted(
                    rng.choice(image_ids[2:],
                               size=min(samples_per_video, len(prob_per_image.to_numpy().nonzero()[0])),
                               p=prob_per_image,
                               replace=False)
                )
        image_ids_left = list(set(image_ids[2:]) - set(hard_chosen_image_ids))
        chosen_image_ids = sorted(
            list(rng.choice(image_ids_left,
                            size=min(samples_per_video-len(hard_chosen_image_ids), len(image_ids_left)),
                            replace=False
                            )
                 ) + hard_chosen_image_ids
        )
        for img_id in chosen_image_ids:
            sample_idx = img_id * 1000
            idx = image_ids.index(img_id)
            detections = video_detections[video_detections.image_id == img_id].copy()
            if len(detections) == 0:
                continue
            detections["detection"] = True
            detections["sample_id"] = sample_idx
            for track_id, track_detections in video_detections.groupby("track_id"):
                if track_id >= 0:
                    # Only create a tracklet when it is valid
                    tracklet_age = 0
                    tracklet_length = max_length
                    tracklet_image_ids = image_ids[max(0, idx - tracklet_length):idx]
                    tracklet = track_detections.loc[track_detections.image_id.isin(tracklet_image_ids)]
                    tracklet = tracklet.copy()
                    tracklet["sample_id"] = sample_idx
                    tracklet["detection"] = False
                else:
                    tracklet = pd.DataFrame(columns=detections.columns)
                detection = detections[detections.track_id == track_id]
                assert len(detection) <= 1
                if len(tracklet) == 0 or len(detection) == 0:
                    continue
                if len(detection) > 0 and hasattr(detection, "id_switch"):
                    id_switch = int(detection.id_switch.item())
                else:
                    id_switch = 0
                samples.append(
                    {
                        "video_id": str(video_id),
                        "image_id": int(img_id),
                        "file_path": metadatas.loc[img_id].file_path,
                        "detections": [x for x in tracklet.index]
                                      + [x for x in detection.index],
                        "to_match": [0 for x in tracklet.index] + [1 for x in detection.index],
                        "id_switch": id_switch,
                    }
                )
    return samples
