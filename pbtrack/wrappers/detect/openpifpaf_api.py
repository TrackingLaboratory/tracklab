import sys
import torch

from PIL import Image
from omegaconf.listconfig import ListConfig

from pbtrack.core.detector import Detector
from pbtrack.core.datastruct import Detection
from pbtrack.utils.images import cv2_load_image
from pbtrack.utils.coordinates import openpifpaf_kp_to_bbox

from pathlib import Path
import pbtrack

root_dir = Path(pbtrack.__file__).parents[1]
sys.path.append(
    str((root_dir / "plugins/detect/openpifpaf/src").resolve())
)  # FIXME : ugly
import openpifpaf

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


class OpenPifPaf(Detector):
    collate_fn = collate_images_anns_meta

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
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
    def preprocess(self, img_meta):
        image = Image.fromarray(cv2_load_image(img_meta.file_path))
        processed_image, anns, meta = self.pifpaf_preprocess(image, [], {})
        return processed_image, anns, meta

    @torch.no_grad()
    def process(self, preprocessed_batch, metadatas):
        processed_image_batch, _, metas = preprocessed_batch
        pred_batch = self.processor.batch(
            self.model, processed_image_batch, device=self.device
        )
        detections = []
        for predictions, meta, (_, metadata) in zip(
            pred_batch, metas, metadatas.iterrows()
        ):
            for prediction in predictions:
                prediction = prediction.inverse_transform(meta)
                detections.append(
                    Detection.create(
                        image_id=metadata.id,
                        id=self.id,
                        keypoints_xyc=prediction.data,
                        bbox_ltwh=openpifpaf_kp_to_bbox(prediction.data),
                    )
                )
                self.id += 1
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

    def _hydra_to_argv(self, cfg):
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
