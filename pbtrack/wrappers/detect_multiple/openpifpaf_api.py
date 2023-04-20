import sys
import torch
import pandas as pd
from PIL import Image
from omegaconf.listconfig import ListConfig

import openpifpaf

from pbtrack import MultiDetector
from pbtrack.utils.images import cv2_load_image
from pbtrack.utils.coordinates import sanitize_keypoints, generate_bbox_from_keypoints

import logging

log = logging.getLogger(__name__)


def collate_images_anns_meta(batch):
    idxs = [b[0] for b in batch]
    batch = [b[1] for b in batch]
    anns = [b[-2] for b in batch]
    metas = [b[-1] for b in batch]

    processed_images = torch.utils.data.dataloader.default_collate(
        [b[0] for b in batch]
    )
    idxs = torch.utils.data.dataloader.default_collate(idxs)
    return idxs, (processed_images, anns, metas)


class OpenPifPaf(MultiDetector):
    collate_fn = collate_images_anns_meta

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        self.id = 0

        if cfg.predict.checkpoint:
            old_argv = sys.argv
            sys.argv = self._hydra_to_argv(cfg.predict)
            openpifpaf.predict.pbtrack_cli()
            predictor = openpifpaf.Predictor()
            sys.argv = old_argv

            self.model = predictor.model
            self.pifpaf_preprocess = predictor.preprocess
            self.processor = predictor.processor
            log.info(
                f"Loaded detection model from checkpoint: {cfg.predict.checkpoint}"
            )

    @torch.no_grad()
    def preprocess(self, metadata: pd.Series):
        image = Image.fromarray(cv2_load_image(metadata.file_path))
        processed_image, anns, meta = self.pifpaf_preprocess(image, [], {})
        return processed_image, anns, meta

    @torch.no_grad()
    def process(self, batch, metadatas: pd.DataFrame, return_fields=False):
        processed_image_batch, _, metas = batch
        pred_batch, fields_batch = self.processor.batch(
            self.model, processed_image_batch, device=self.device
        )
        detections = []
        for predictions, meta, (_, metadata) in zip(
            pred_batch, metas, metadatas.iterrows()
        ):
            for prediction in predictions:
                prediction = prediction.inverse_transform(meta)
                keypoints = sanitize_keypoints(prediction.data, meta["width_height"])
                bbox = generate_bbox_from_keypoints(
                    keypoints[keypoints[:, 2] > 0],
                    self.cfg.bbox.extension_factor,
                    meta["width_height"],
                )
                detections.append(
                    pd.Series(
                        dict(
                            image_id=metadata.name,
                            keypoints_xyc=keypoints,
                            keypoints_conf=prediction.score,
                            bbox_ltwh=bbox,
                            bbox_conf=prediction.score,
                            video_id=metadata.video_id,
                            category_id=1,  # `person` class in posetrack
                        ),
                        name=self.id,
                    )
                )
                self.id += 1
        if return_fields:
            return detections, fields_batch
        else:
            return detections

    def train(self):
        old_argv = sys.argv
        sys.argv = self._hydra_to_argv(self.cfg.train)
        log.info(f"Starting training of the detection model")
        self.cfg.predict.checkpoint = openpifpaf.train.main()
        sys.argv = self._hydra_to_argv(self.cfg.predict)
        openpifpaf.predict.pbtrack_cli()
        predictor = openpifpaf.Predictor()
        sys.argv = old_argv

        self.model = predictor.model
        self.pifpaf_preprocess = predictor.preprocess
        self.processor = predictor.processor
        log.info(
            f"Loaded trained detection model from file: {self.cfg.predict.checkpoint}"
        )

    @staticmethod
    def _hydra_to_argv(cfg):
        new_argv = ["argv_from_hydra"]
        for k, v in cfg.items():
            new_arg = f"--{str(k)}"
            if isinstance(v, ListConfig):
                new_argv.append(new_arg)
                for item in v:
                    new_argv.append(f"{str(item)}")
            elif v is not None:
                new_arg += f"={str(v)}"
                new_argv.append(new_arg)
            else:
                new_argv.append(new_arg)
        return new_argv
