# import cv2
# import torch
# import pandas as pd
# from mmpose.datasets import Compose
#
# from tracklab.pipeline import ImageLevelModule
# from tracklab.utils.coordinates import ltrb_to_ltwh
# from tracklab.utils.openmmlab import get_checkpoint
#
# import mmcv
# from mmengine.dataset.utils import default_collate as collate
# # from mmcv.parallel import collate, scatter
# from mmdet.apis import init_detector
#
# import logging
#
# log = logging.getLogger(__name__)
# mmcv.collect_env()
#
#
# def mmdet_collate(batch):
#     return collate(batch, len(batch))
#
#
# class MMDetection(ImageLevelModule):
#     collate_fn = mmdet_collate
#     output_columns = [
#         "image_id",
#         "video_id",
#         "category_id",
#         "bbox_ltwh",
#         "bbox_conf",
#     ]
#
#     def __init__(self, cfg, device, batch_size):
#         super().__init__(batch_size)
#         get_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
#         self.model = init_detector(cfg.path_to_config, cfg.path_to_checkpoint, device)
#         self.id = 0
#         self.device = device
#
#         cfg = self.model.cfg
#         self.cfg = cfg.copy()  # FIXME check if needed
#         # set loading pipeline type
#         self.cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
#         # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
#         self.test_pipeline = Compose(cfg.data.test.pipeline)
#
#     @torch.no_grad()
#     def preprocess(self, metadata: pd.Series):
#         image = cv2.imread(metadata.file_path)  # BGR not RGB !
#         data = {
#             "img": image,
#         }
#         return self.test_pipeline(data)
#
#     @torch.no_grad()
#     def process(self, batch, metadatas: pd.DataFrame):
#         # just get the actual data from DataContainer
#         batch["img_metas"] = [img_metas.data[0] for img_metas in batch["img_metas"]]
#         batch["img"] = [img.data[0] for img in batch["img"]]
#         batch = batch.to(self.device)
#         results = self.model(return_loss=False, rescale=True, **batch)
#         shapes = [(x["ori_shape"][1], x["ori_shape"][0]) for x in batch["img_metas"][0]]
#         detections = []
#         for predictions, image_shape, (_, metadata) in zip(
#             results, shapes, metadatas.iterrows()
#         ):
#             for prediction in predictions[0]:  # only check for 'person' class
#                 if prediction[4] >= self.cfg.min_confidence:
#                     detections.append(
#                         pd.Series(
#                             dict(
#                                 image_id=metadata.name,
#                                 bbox_ltwh=ltrb_to_ltwh(prediction[:4], image_shape),
#                                 bbox_conf=prediction[4],
#                                 video_id=metadata.video_id,
#                                 category_id=1,  # `person` class in posetrack
#                             ),
#                             name=self.id,
#                         )
#                     )
#                     self.id += 1
#         return detections
