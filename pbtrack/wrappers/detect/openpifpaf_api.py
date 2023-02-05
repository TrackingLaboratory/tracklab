import sys
import torch
import numpy as np

from PIL import Image
from omegaconf.listconfig import ListConfig

from pbtrack.core.detector import Detector
from pbtrack.core.datastruct import Detection
from pbtrack.utils.images import cv2_load_image
from pbtrack.utils.coordinates import round_bbox_coordinates

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
        image = Image.fromarray(
            cv2_load_image(img_meta.file_path)
        )  # TODO Image should be loaded in track_engine (could be loaded differently, from mp4 for instance)
        processed_image, anns, meta = self.pifpaf_preprocess(image, [], {})
        return processed_image, anns, meta

    @torch.no_grad()
    def process(self, preprocessed_batch, metadatas, return_fields=False):
        processed_image_batch, _, metas = preprocessed_batch
        pred_batch, fields_batch = self.processor.batch(
            self.model, processed_image_batch, device=self.device
        )
        detections = []
        for predictions, meta, (_, metadata) in zip(
            pred_batch, metas, metadatas.iterrows()
        ):
            for prediction in predictions:
                prediction = prediction.inverse_transform(meta)
                keypoints = prediction.data

                w, h = meta["width_height"]
                keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
                keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)
                keypoints[:, 2] = np.clip(keypoints[:, 2], 0, 1)

                bbox_ltrb = self.keypoints_to_bbox(keypoints)
                bbox_ltrb = round_bbox_coordinates(bbox_ltrb)

                bbox_ltrb[0] = np.clip(bbox_ltrb[0], 0, w - 1)
                bbox_ltrb[1] = np.clip(bbox_ltrb[1], 0, h - 1)
                bbox_ltrb[2] = np.clip(bbox_ltrb[2], 0, w - 1)
                bbox_ltrb[3] = np.clip(bbox_ltrb[3], 0, h - 1)
                bbox_ltwh = np.array(
                    [
                        bbox_ltrb[0],
                        bbox_ltrb[1],
                        bbox_ltrb[2] - bbox_ltrb[0],
                        bbox_ltrb[3] - bbox_ltrb[1],
                    ]
                )

                detections.append(
                    Detection.create(
                        image_id=metadata.id,
                        id=self.id,
                        keypoints_xyc=keypoints,
                        bbox_ltwh=bbox_ltwh,
                        bbox_c=np.mean(keypoints[:, 2], axis=0),
                        video_id=metadata.video_id,
                    )
                )
                self.id += 1
        if return_fields:
            return detections, fields_batch
        else:
            return detections

    def keypoints_to_bbox(self, keypoints):
        keypoints = keypoints[keypoints[:, 2] > 0]
        lt = np.amin(keypoints[:, :2], axis=0)
        rb = np.amax(keypoints[:, :2], axis=0)
        bbox_w = rb[0] - lt[0]
        bbox_h = rb[1] - lt[1]
        lt[0] -= bbox_w * self.cfg.bbox.left_right_extend_factor
        rb[0] += bbox_w * self.cfg.bbox.left_right_extend_factor
        lt[1] -= bbox_h * self.cfg.bbox.top_extend_factor
        rb[1] += bbox_h * self.cfg.bbox.bottom_extend_factor
        return np.array([lt[0], lt[1], rb[0], rb[1]])

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
