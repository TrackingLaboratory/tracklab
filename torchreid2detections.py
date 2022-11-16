import sys
import numpy as np
import torch

from reconnaissance.datasets.posetrack21_reid import PoseTrack21ReID
from scripts.main import build_torchreid_model_engine, build_config
from reconnaissance.utils.coordinates import kp_img_to_kp_bbox, rescale_keypoints
from tools.feature_extractor import FeatureExtractor
from torchreid.data.data_augmentation import CocoToSixBodyMasks
from torchreid.utils.imagetools import build_gaussian_heatmaps

sys.path.append('bpbreid')
import torchreid
# need that line to not break import of torchreid ('from torchreid... import ...') inside the bpbreid.torchreid module
# to remove the 'from torchreid... import ...' error 'Unresolved reference 'torchreid' in PyCharm, right click
# on 'bpbreid' folder, then choose 'Mark Directory as' -> 'Sources root'
from bpbreid.scripts.default_config import engine_run_kwargs


def configure_dataset_class(clazz, **ext_kwargs):
    """
    Wrapper function to provide the class with args external to torchreid
    """
    class ClazzWrapper(clazz):
        def __init__(self, **kwargs):
            self.__name__ = clazz.__name__
            super(ClazzWrapper, self).__init__(**{**kwargs, **ext_kwargs})

    ClazzWrapper.__name__ = clazz.__name__

    return ClazzWrapper


class Torchreid2detections:
    """
    TODO:
        why bbox move after strongsort?
        training
        batch process
        save config + commit hash with model weights
        model download from URL: HRNet etc
        save folder: uniform with reconnaissance
        wandb support
    """
    def __init__(self, device, save_path, config_path, model_pose):
        config = {
            'crop_dim': (384, 128),
            'datasets_root': '~/datasets/other',
            'pose_model': model_pose
        }
        torchreid.data.register_image_dataset("posetrack21_reid", configure_dataset_class(PoseTrack21ReID, **config), "pt21")
        self.cfg = build_config(config_file=config_path)  # TODO support command line args as well?
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg.data.save_dir = save_path
        self.device = device
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = not self.cfg.test.evaluate
        self.feature_extractor = None
        self.model = None
        self.transform = CocoToSixBodyMasks()

    def _image2input(self, image): # Tensor RGB (1, 3, H, W)
        assert 1 == image.shape[0], "Test batch size should be 1"
        input = image[0].cpu().numpy() # -> (3, H, W)
        input = np.transpose(input, (1, 2, 0)) # -> (H, W, 3)
        input = input*255.0
        input = input.astype(np.uint8) # -> to uint8
        return input

    def run(self, detections, image):
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.device,
                image_size=(self.cfg.data.height, self.cfg.data.width),
                model=self.model
            )
        mask_w, mask_h = 32, 64
        im_crops = []
        image = self._image2input(image)
        all_masks = []
        for i, bbox in enumerate(detections.Bboxes):
            l, t, r, b = bbox.astype(int)
            crop = image[t:b, l:r]
            im_crops.append(crop)
            keypoints = np.array(detections.Poses[i])
            bbox_ltwh = np.array([l, t, r - l, b - t])
            kp_xyc_bbox = kp_img_to_kp_bbox(keypoints, bbox_ltwh)
            kp_xyc_mask = rescale_keypoints(kp_xyc_bbox, (bbox_ltwh[2], bbox_ltwh[3]), (mask_w, mask_h))
            pixels_parts_probabilities = build_gaussian_heatmaps(kp_xyc_mask, mask_w, mask_h)
            all_masks.append(pixels_parts_probabilities)


        if im_crops:
            external_parts_masks = np.stack(all_masks, axis=0)
            embeddings, visibility_scores, body_masks, _ = self.feature_extractor(im_crops, external_parts_masks=external_parts_masks)
            detections.add_reid_features(embeddings)
            detections.add_visibility_scores(visibility_scores)
            detections.add_body_masks(body_masks)
        return detections

    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))

    def _xywh_to_xyxy(self, bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2
