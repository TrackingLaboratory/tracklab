import cv2
import torch
import pandas as pd

from pbtrack import MultiDetector
from pbtrack.utils.coordinates import ltrb_to_ltwh
from pbtrack.utils.openmmlab import get_checkpoint

import mmcv
from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

import logging

log = logging.getLogger(__name__)
mmcv.collect_env()


def mmdet_collate(batch):
    return collate(batch, len(batch))


@torch.no_grad()
class MMDetection(MultiDetector):
    collate_fn = mmdet_collate

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        get_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.model = init_detector(cfg.path_to_config, cfg.path_to_checkpoint, device)
        self.id = 0

        cfg = self.model.cfg
        cfg = cfg.copy()  # FIXME check if needed
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        self.test_pipeline = Compose(cfg.data.test.pipeline)

    def preprocess(self, metadata: pd.Series):
        image = cv2.imread(metadata.file_path)  # BGR not RGB !
        data = {
            "img": image,
        }
        return self.test_pipeline(data)

    def process(self, batch, metadatas: pd.DataFrame):
        # just get the actual data from DataContainer
        batch["img_metas"] = [img_metas.data[0] for img_metas in batch["img_metas"]]
        batch["img"] = [img.data[0] for img in batch["img"]]
        batch = scatter(batch, [self.device])[0]
        results = self.model(return_loss=False, rescale=True, **batch)
        shapes = [(x["ori_shape"][1], x["ori_shape"][0]) for x in batch["img_metas"][0]]
        detections = []
        for predictions, image_shape, (_, metadata) in zip(
            results, shapes, metadatas.iterrows()
        ):
            for prediction in predictions[0]:  # only check for 'person' class
                if prediction[4] >= self.cfg.min_confidence:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(prediction[:4], image_shape),
                                bbox_conf=prediction[4],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections
