import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from hydra.utils import instantiate
from pathlib import Path
from dd_sort import normalize_kps, normalize_bbox, PairsStatistics

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.cv2 import cv2_load_image

log = logging.getLogger(__name__)


class DDSORT(ImageLevelModule):
    input_columns = ["bbox_ltwh", "bbox_conf", "keypoints_xyc", "visibility_scores", "embeddings"]
    output_columns = [
        "track_id",
    ]

    def __init__(
        self,
        simformer,
        device,
        datamodule,
        train_cfg,
        ddsort,
        det_filter_cfg,
        checkpoint_path,
        tracking_dataset,
        override_cfg=None,
        training_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(batch_size=1)
        self.device = device
        if override_cfg is None:
            override_cfg = {}
        self.tracking_dataset = tracking_dataset
        self.simformer = instantiate(simformer, _recursive_=False)
        self.train_cfg = train_cfg
        self.ddsort = ddsort
        self.override_cfg = override_cfg
        self.datamodule_cfg = datamodule
        self.training_enabled = training_enabled

        if checkpoint_path:
            self.simformer = type(self.simformer).load_from_checkpoint(
                checkpoint_path, map_location=self.device, **override_cfg
            )
            log.info(f"Loading simformer checkpoint from `{checkpoint_path}`.")

        self.min_bbox_threshold = det_filter_cfg.min_bbox_threshold
        self.vis_keypoint_threshold = det_filter_cfg.vis_keypoint_threshold
        self.min_vis_keypoints = det_filter_cfg.min_vis_keypoints

        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.tracker = [
            instantiate(v, self.simformer.to(self.device))  # Instantiate DDSort or DDSortByteTracker
            for k, v in self.ddsort.items()
            if v["_enabled_"]
        ][0]

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        keep_flags = []
        for det_id, detection in detections.iterrows():
            if 'keypoints_xyc' in detection and self.vis_keypoint_threshold is not None:
                detection["keypoints_xyc"][
                    detection["keypoints_xyc"][:, 2] < self.vis_keypoint_threshold,
                    2,
                ] = 0
                keep = (detection["bbox_conf"] >= self.min_bbox_threshold) and (
                    (detection["keypoints_xyc"][:, 2] != 0).sum() >= self.min_vis_keypoints)
            else:
                keep = (detection["bbox_conf"] >= self.min_bbox_threshold)
            keep_flags.append(keep)
        keep_flags = np.array(keep_flags)
        return {"keep_flags": keep_flags}

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        keep = batch["keep_flags"][0]
        if not len(keep):
            return []
        pbtrack_ids = torch.tensor(detections.index, dtype=torch.int32)[keep]
        features = {}
        for feature_name in self.input_columns:
            if feature_name in detections:
                features[feature_name] = torch.tensor(np.stack(detections[feature_name])[list(keep)], dtype=torch.float32).unsqueeze(0)
        image = cv2_load_image(metadatas['file_path'].values[0])
        image_shape = torch.tensor(image.shape[:-1][::-1])
        features["bbox_ltwh"] = normalize_bbox(features["bbox_ltwh"], image_shape)
        if "keypoints_xyc" in features:
            features["keypoints_xyc"] = normalize_kps(features["keypoints_xyc"], image_shape)
        image_id = int(metadatas.index[0])
        results = self.tracker.update(features, pbtrack_ids, image_id, image)
        if results:
            results = pd.DataFrame(results)
            results.set_index("pbtrack_id", inplace=True, drop=True)
            assert set(results.index).issubset(
                detections.index
            ), "Mismatch of indexes during the tracking. The results should match the detections."
            return results
        else:
            return []

    def train(self, tracking_dataset, pipeline, *args, **kwargs):
        self.datamodule = instantiate(self.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=pipeline)
        save_best_auroc = pl.callbacks.ModelCheckpoint(
            monitor="val/sim_auroc",
            mode="max",
            filename="epoch={epoch}-sim_auroc={val/sim_auroc:.4f}-loss={val/loss:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        save_best_loss = pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            filename="epoch={epoch}-sim_auroc={val/sim_auroc:.4f}-loss={val/loss:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        callbacks = [
            save_best_auroc,
            save_best_loss,
            pl.callbacks.LearningRateMonitor(),
            # DisplayBatchSamples(self.datamodule.tracking_sets, True, enabled_steps=["train", "val", "test"]),
            PairsStatistics(),
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=5, mode="min", check_on_train_epoch_end=False),
        ]
        if self.train_cfg.use_rich:
            callbacks.append(pl.callbacks.RichProgressBar())
        if self.train_cfg.use_wandb:
            logger = pl.loggers.WandbLogger(project="DDSort", entity="pbtrack", resume=True)
        else:
            logger = pl.loggers.WandbLogger(project="DDSort", entity="pbtrack", offline=True)
        tr_cfg = self.train_cfg.pl_trainer
        trainer = pl.Trainer(
            max_epochs=tr_cfg.max_epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator="auto",
            num_sanity_val_steps=tr_cfg.num_sanity_val_steps,
            fast_dev_run=tr_cfg.fast_dev_run,
            precision=tr_cfg.precision,
            gradient_clip_val=tr_cfg.gradient_clip_val,
            accumulate_grad_batches=tr_cfg.accumulate_grad_batches,
            log_every_n_steps=tr_cfg.log_every_n_steps,
            check_val_every_n_epoch=tr_cfg.check_val_every_n_epochs,
            val_check_interval=tr_cfg.val_check_interval,
            enable_progress_bar=tr_cfg.enable_progress_bar,
            profiler=tr_cfg.profiler,
            enable_model_summary=tr_cfg.enable_model_summary,
        )
        if not self.train_cfg.evaluate_only:
            trainer.fit(self.simformer, self.datamodule)
            trainer.save_checkpoint("default_checkpoint.ckpt")
            checkpoint_path = save_best_loss.best_model_path if save_best_loss.best_model_path else save_best_loss.last_model_path
            if checkpoint_path:
                log.info(f"Loading simformer checkpoint from `{Path(checkpoint_path).resolve()}`.")
                type(self.simformer).load_from_checkpoint(checkpoint_path, map_location=self.device)
            else:
                log.warning("No simformer checkpoint found.")
        trainer.validate(self.simformer, self.datamodule)
