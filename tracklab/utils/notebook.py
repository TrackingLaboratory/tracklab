import os
import copy
import torch

from pathlib import Path
from collections import namedtuple
from omegaconf import OmegaConf, read_write, open_dict

# from hydra import initialize_config_module as init_hydra, compose
from hydra import initialize_config_dir as init_hydra, compose
from hydra.utils import instantiate
from hydra.core.utils import configure_log

from tracklab.datastruct import TrackerState

TrackEngine = namedtuple(
    "TrackEngine", ["cfg", "engine", "state", "evaluator", "dataset"]
)


def _save_config(cfg, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(output_dir / filename), "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg))


def load_from_overrides(overrides=[]) -> TrackEngine:
    """Load everything as in main(), but from notebook.

    Use with ::
        import os
        import sys
        sys.path.append(os.getcwd()+"/..")
        from tracklab.utils import load_from_overrides

        load_from_overrides(["train_detect=false"])

    Args:
        overrides: list of strings as used in hydra commandline

    Returns:
        track_engine: a tuple containing : cfg, engine, state, evaluator, dataset
    """
    if hasattr(load_from_overrides, "orig_dir"):
        os.chdir(load_from_overrides.orig_dir)
    overrides += ["experiment_name=notebook"]
    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO support Mac chips
    with init_hydra(config_dir=str(Path("../configs").resolve()), version_base=None):
        cfg = compose(
            config_name="config", overrides=overrides, return_hydra_config=True
        )
    task_cfg = copy.deepcopy(cfg)
    with read_write(task_cfg):
        with open_dict(task_cfg):
            del task_cfg["hydra"]
    # Manage hydra
    output_dir = str(OmegaConf.select(cfg, "hydra.run.dir"))

    with read_write(cfg.hydra.runtime):
        with open_dict(cfg.hydra.runtime):
            cfg.hydra.runtime.output_dir = os.path.abspath(output_dir)
    Path(str(output_dir)).mkdir(parents=True, exist_ok=True)
    configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    _chdir = cfg.hydra.job.chdir
    if _chdir:
        load_from_overrides.orig_dir = os.getcwd()
        os.chdir(output_dir)
    hydra_cfg = cfg.hydra
    if cfg.hydra.output_subdir is not None:
        hydra_output = Path(cfg.hydra.runtime.output_dir) / Path(
            cfg.hydra.output_subdir
        )
        _save_config(task_cfg, "config.yaml", hydra_output)
        _save_config(hydra_cfg, "hydra.yaml", hydra_output)
        _save_config(cfg.hydra.overrides.task, "overrides.yaml", hydra_output)

    tracking_dataset = instantiate(cfg.dataset)
    model_detect = instantiate(cfg.detect, device=device)
    reid_model = instantiate(
        cfg.reid,
        tracking_dataset=tracking_dataset,
        device=device,
        model_detect=model_detect,
    )
    track_model = instantiate(cfg.track, device=device)
    evaluator = instantiate(cfg.eval)
    vis_engine = instantiate(cfg.visualization)

    val_state = TrackerState(tracking_dataset.sets['val'])

    tracking_engine = instantiate(
        cfg.engine,
        model_detect=model_detect,
        reid_model=reid_model,
        track_model=track_model,
        tracker_state=val_state,
        vis_engine=vis_engine,
    )

    return TrackEngine(cfg, tracking_engine, val_state, evaluator, tracking_dataset)
