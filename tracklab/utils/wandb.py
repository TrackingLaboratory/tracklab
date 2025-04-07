from typing import Mapping, Sequence
try:
    import wandb
except:
    pass
from omegaconf import OmegaConf
import pandas as pd

import logging

logger = logging.getLogger(__name__)

# FIXME not sure it is the right to do that. It is annoying to update this every time we add a new config
keep_dict = {
    "dataset": ["dataset_path", "nframes", "nvid", "vids_dict"],
    "detect_multiple": [
        "min_confidence",
        "path_to_config",
        "path_to_checkpoint",
        "instance_min_confidence",
        "keypoint_min_confidence",
        "bbox",
        "predict",
        "train",
    ],
    "detect_single": [
        "min_keypoints_score",
        "min_keypoints_confidence",
        "path_to_config",
        "path_to_checkpoint",
        "bbox",
        "predict",
        "train",
    ],
    "eval": ["mot"],
    "reid": ["data", "loss", "model", "sampler", "test", "train", "dataset"],
    "track": True,
}


def normalize_subdict(subdict):
    if "_target_" in subdict:
        subdict["target"] = subdict.pop("_target_")
    if "cfg" in subdict:
        for k, v in subdict["cfg"].items():
            subdict[k] = v
        del subdict["cfg"]
    return subdict


def init(cfg):
    global use_wandb
    use_wandb = cfg.use_wandb
    if use_wandb:
        kwargs = {}
        if "wandb" in cfg:
            kwargs = cfg.wandb
        cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg["experiment_name"], config=cfg, **kwargs)


def log_metric(res_dict, name, video_dict=None):
    if use_wandb:
        try:
            wandb.log(
                {f"{name}/{k}": v for k, v in res_dict.items()},
                step=0,
            )
            if video_dict is not None:
                video_df = pd.DataFrame.from_dict(video_dict, orient="index")
                video_df.insert(0, "video", video_df.index)
                wandb.log({f"{name}/videos": video_df}, step=0)
        except wandb.Error:
            logger.warning("Wandb error, skipping logging")
            pass


def log(res_dict):
    if use_wandb:
        try:
            wandb.log(res_dict)
        except wandb.Error:
            logger.warning("Wandb error, skipping logging")
            pass


def apply_recursively(d, f=lambda v: v, filter=lambda k, v: True, always_filter=False):
    """
    Apply a function to leaf values of a dict recursively and/or filter dict

    Args:
        f: function taking the value of a leaf as argument and returning
           a transformation of that value
        filter: condition to apply f to only (sub-)branches of the tree
        always_filter: if true filter sub-branches, if false only filter leaves.
    Returns:
        transformed and filtered dict
    """
    for k, v in d.items():
        if isinstance(v, Mapping):
            if always_filter:
                d[k] = apply_recursively(v, f, filter) if filter(k, v) else v
            else:
                d[k] = apply_recursively(v, f, filter, always_filter)
        elif isinstance(v, str):
            d[k] = f(v) if filter(k, v) else v
        elif isinstance(v, Sequence):
            if len(v) == 0:
                d[k] = v
            elif isinstance(v[0], Mapping):
                d[k] = [apply_recursively(val, f, filter) for val in v]
            else:
                d[k] = [f(val) if filter(k, val) else val for val in v]
        else:
            d[k] = f(v) if filter(k, v) else v
    return d


def finish():
    if use_wandb:
        wandb.finish()
