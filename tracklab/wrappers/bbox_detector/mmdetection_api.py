from pathlib import Path
from typing import Any

import pandas as pd
import torch
from mim import get_model_info
from mim.utils import get_installed_path
from mmcv import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmengine.dataset import default_collate

from tracklab.pipeline import ImageLevelModule
from mmdet.apis.inference import init_detector

from tracklab.utils import ltrb_to_ltwh
from tracklab.utils.openmmlab import get_checkpoint


class MMDetection(ImageLevelModule):
    collate_fn = default_collate
    input_columns = []
    output_columns = [
        "image_id", "video_id", "category_id", "bbox_ltwh", "bbox_conf"
    ]

    def __init__(self, config_name, path_to_checkpoint, device, batch_size,
                 min_confidence, **kwargs):
        super().__init__(batch_size)
        self.device = device
        self.min_confidence = min_confidence
        model_df = get_model_info(package="mmdet", configs=[config_name])
        if len(model_df) != 1:
            raise ValueError(f"Multiple values found for config_name: {config_name}")

        download_url = model_df.weight.item()
        package_path = Path(get_installed_path("mmdet"))
        path_to_config = package_path / ".mim" / model_df.config.item()
        get_checkpoint(path_to_checkpoint, download_url)
        self.model = init_detector(str(path_to_config), path_to_checkpoint, device=device)
        self.test_pipeline = get_test_pipeline_cfg(self.model.cfg.copy())
        self.test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(self.test_pipeline)
        self.current_id = 0

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.test_pipeline(dict(img=image, img_id=0))

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        results = self.model.test_step(batch)
        img_metas = batch["data_samples"]
        shapes = [(x.ori_shape[1], x.ori_shape[0]) for x in batch["data_samples"]]
        detections = []
        for preds, image_shape, (_, metadata) in zip(results, shapes, metadatas.iterrows()):
            instances = preds.pred_instances
            for score, bbox, label in zip(instances.scores, instances.bboxes, instances.labels):
                if score < self.min_confidence or label != 0:
                    continue
                detections.append(
                    pd.Series(dict(
                        image_id=metadata.name,
                        video_id=metadata.video_id,
                        bbox_ltwh=ltrb_to_ltwh(bbox.cpu().numpy(), image_shape),
                        bbox_conf=float(score.item()),
                        category_id=1,  # 'person' class
                    ), name=self.current_id)
                )
                self.current_id += 1

        return pd.DataFrame(detections)

