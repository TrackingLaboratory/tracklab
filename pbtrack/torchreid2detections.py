import sys
import numpy as np
import torch

from pbtrack.datasets.posetrack21_reid import PoseTrack21ReID
from pbtrack.utils.coordinates import kp_img_to_kp_bbox, rescale_keypoints

from modules.reid.bpbreid.scripts.main import build_config, build_torchreid_model_engine
from modules.reid.bpbreid.tools.feature_extractor import FeatureExtractor
from modules.reid.bpbreid.torchreid.data.data_augmentation.coco_keypoints_transforms import CocoToSixBodyMasks
from modules.reid.bpbreid.torchreid.utils.imagetools import build_gaussian_heatmaps

sys.path.append('modules/reid/bpbreid')  # FIXME ugly
sys.path.append('modules/reid')  # FIXME ugly
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
    def __init__(self, device, save_path, config_path, model_pose, job_id):
        config = {
            'crop_dim': (384, 128),
            'datasets_root': '~/datasets/other',
            'pose_model': model_pose
        }
        torchreid.data.register_image_dataset("posetrack21_reid", configure_dataset_class(PoseTrack21ReID, **config), "pt21")
        self.cfg = build_config(config_file=config_path)  # TODO support command line args as well?
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
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

    def run(self, detections, data):
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
        image = self._image2input(data['image'].unsqueeze(0))  # FIXME
        all_masks = []
        for i, detection in enumerate(detections):
            bbox = detection.bbox
            pose = detection.keypoints
            l = int(bbox.x)
            t = int(bbox.y)
            r = int(bbox.x+bbox.w)
            b = int(bbox.y+bbox.h)
            crop = image[t:b, l:r]
            im_crops.append(crop)
            keypoints = np.array([[kp.x, kp.y, kp.conf] for kp in detection.keypoints])
            bbox_ltwh = np.array([l, t, r - l, b - t])
            kp_xyc_bbox = kp_img_to_kp_bbox(keypoints, bbox_ltwh)
            kp_xyc_mask = rescale_keypoints(kp_xyc_bbox, (bbox_ltwh[2], bbox_ltwh[3]), (mask_w, mask_h))
            pixels_parts_probabilities = build_gaussian_heatmaps(kp_xyc_mask, mask_w, mask_h)
            all_masks.append(pixels_parts_probabilities)

        if im_crops:
            external_parts_masks = np.stack(all_masks, axis=0)
            embeddings, visibility_scores, body_masks, _ = self.feature_extractor(im_crops, external_parts_masks=external_parts_masks)
            for i, detection in enumerate(detections):
                detection.reid_features = embeddings[i]
                detection.visibility_score = visibility_scores[i]
                detection.body_mask = body_masks[i]
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
