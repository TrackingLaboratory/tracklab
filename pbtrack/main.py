import os
import torch
import hydra
from pbtrack.utils import monkeypatch_hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pbtrack.datastruct import TrackerState
from pbtrack.pipeline import Pipeline
from pbtrack.utils import wandb

import logging

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )  # FIXME : why are we using too much file descriptors ?


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # For Hydra and Slurm compatibility
    set_sharing_strategy()  # Do not touch

    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO support Mac chips
    log.info(f"Using device: '{device}'.")

    wandb.init(cfg)
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))

    # Initiate all the instances
    tracking_dataset = instantiate(cfg.dataset)
    modules = []
    for name in cfg.pipeline:
        module = cfg.modules[name]
        inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
        modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    evaluator = instantiate(cfg.eval)

    # FIXME, je pense qu'il faut repenser l'entrainement dans ce script
    # On peut pas entrainer 3 modèles dans un script quoi qu'il arrive
    # Il faut que l'entrainement soit fait dans un script propre à la librairie
    for module in modules:
        if hasattr(module, "train"):
            log.info(f"Not actually training {module.name}")
            pass  # FIXME : really train if they want

    if cfg.test_tracking:
        log.info(f"Starting tracking operation on {cfg.eval.test_set} set.")
        if cfg.eval.test_set == "train":
            tracking_set = tracking_dataset.train_set
        elif cfg.eval.test_set == "val":
            tracking_set = tracking_dataset.val_set
        else:
            tracking_set = tracking_dataset.test_set
        tracker_state = TrackerState(tracking_set, modules=pipeline, **cfg.state)
        # Run tracking and visualization
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )
        tracking_engine.track_dataset()
        # Evaluation
        if cfg.dataset.nframes == -1:
            if tracker_state.detections_gt is not None:
                log.info("Starting evaluation.")
                evaluator.run(tracker_state)
            else:
                log.warning(
                    "Skipping evaluation because there's no ground truth detection."
                )
        else:
            log.warning(
                "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
                "to -1)"
            )

        if tracker_state.save_file is not None:
            log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    return 0


if __name__ == "__main__":
    main()
