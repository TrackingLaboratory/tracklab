import sys
import torch
from PIL import Image

from pbtrack.core.detector import Detector
from pbtrack.core.datastruct import Detection
from pbtrack.utils.images import cv2_load_image
from pbtrack.utils.coordinates import kp_to_bbox_w_threshold

import pbtrack
from pathlib import Path

root_dir = Path(pbtrack.__file__).parents[1]
sys.path.append(str((root_dir / "plugins/detect/openpifpaf/src").resolve())) # FIXME : ugly
import openpifpaf


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

        old_argv = sys.argv
        sys.argv = hydra_to_argv(cfg)
        openpifpaf.predict.pbtrack_cli()
        predictor = openpifpaf.Predictor()
        sys.argv = old_argv

        self.model = predictor.model
        self.pifpaf_preprocess = predictor.preprocess
        self.processor = predictor.processor

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
                        bbox_ltwh=kp_to_bbox_w_threshold(
                            prediction.data, vis_threshold=0.05
                        ),
                    )
                )
                self.id += 1
        return detections

def hydra_to_argv(cfg):
    new_argv = ["argv_from_hydra"]
    for k, v in cfg.items():
        new_arg = f"--{str(k)}"
        if isinstance(v, list):
            for item in v:
                new_arg += f" {str(item)}"
        elif v is not None:
            new_arg += f"={str(v)}"
        new_argv.append(new_arg)
    return new_argv