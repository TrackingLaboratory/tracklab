import os
import rich.logging
import torch
import hydra
import warnings
import logging

from pbtrack.utils import monkeypatch_hydra  # needed to avoid complex hydra stacktraces when errors occur in "instantiate(...)"
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pbtrack.datastruct import TrackerState
from pbtrack.pipeline import Pipeline
from pbtrack.utils import wandb


os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    device = init_environment(cfg)

    # Instantiate all modules
    tracking_dataset = instantiate(cfg.dataset)
    evaluator = instantiate(cfg.eval)

    modules = []
    for name in cfg.pipeline:
        module = cfg.modules[name]
        inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
        modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    # Train tracking modules
    for module in modules:
        if module.training_enabled:
            # module.train()
            raise NotImplementedError("Module training is not implemented yet.")

    # Test tracking
    if cfg.test_tracking:
        log.info(f"Starting tracking operation on {cfg.eval.test_set} set.")

        # Init tracker state and tracking engine
        tracking_set = getattr(tracking_dataset, cfg.eval.test_set + "_set")
        tracker_state = TrackerState(tracking_set, modules=pipeline, **cfg.state)
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )

        # Run tracking and visualization
        tracking_engine.track_dataset()

        # Evaluation
        evaluate(cfg, evaluator, tracker_state)

        # Save tracker state
        if tracker_state.save_file is not None:
            log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    close_enviroment()

    return 0


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )


def init_environment(cfg):
    # For Hydra and Slurm compatibility
    set_sharing_strategy()  # Do not touch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: '{device}'.")
    wandb.init(cfg)
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_rich:
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.INFO)
    return device


def close_enviroment():
    wandb.finish()


def evaluate(cfg, evaluator, tracker_state):
    if cfg.get("eval_tracking", True) and cfg.dataset.nframes == -1:
        if tracker_state.detections_gt is not None:
            log.info("Starting evaluation.")
            evaluator.run(tracker_state)
        else:
            log.warning(
                "Skipping evaluation because there is no ground truth detection."
            )
    else:
        log.warning(
            "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
            "to -1)"
        )


if __name__ == "__main__":
    main()
