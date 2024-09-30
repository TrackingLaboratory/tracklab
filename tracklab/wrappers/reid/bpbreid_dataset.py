from __future__ import absolute_import, division, print_function

import json

import cv2
import pandas as pd
import torch.nn.functional as F
from math import ceil
from pathlib import Path
from skimage.transform import resize
from torchreid.data import ImageDataset
from torchreid.data.datasets.keypoints_to_masks import kp_img_to_kp_bbox, rescale_keypoints
from torchreid.data.masks_transforms import CocoToEightBodyMasks
from torchreid.utils.imagetools import build_keypoints_heatmaps, build_keypoints_gaussian_heatmaps, \
    build_joints_heatmaps, build_joints_gaussian_heatmaps, gkern
from torchreid.utils.visualization.visualize_query_gallery_rankings import colored_body_parts_overlay, draw_keypoints
from tqdm import tqdm
from yacs.config import CfgNode as CN
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
from PIL import Image
import logging
import os.path as osp
from segment_anything import SamPredictor, sam_model_registry

from tracklab.datastruct import TrackingDataset
from tracklab.utils import clip_bbox_ltwh_to_img_dim, clip_keypoints_to_image, keypoints_in_bbox_coord
from tracklab.wrappers.detect_multiple.openpifpaf_api import OpenPifPaf

log = logging.getLogger(__name__)

# This code os borrowed from Tracklab: https://github.com/TrackingLaboratory/tracklab
# The original purpose of this Tracklab code is to build a ReID dataset from a MOT dataset
# We just copy pasted the relevant parts from Tracklab and adapted them to turn the PoseTrack21 dataset into the
# Occluded-PoseTrack-ReID dataset. This code will generate an new 'reid' folder inside the PoseTrack21 dataset folder,
# containing the ReID dataset, i.e. persons crops, keypoints, and masks.
# This class employs the ground keypoints from PoseTrack21 as prompts, and PifPaf and SAM to generate the pseudo
# human-parsing labels.

class ReidDataset(ImageDataset):
    img_ext = ".jpg"
    masks_ext = ".npy"
    kps_ext = ".json"
    reid_dir = "reid"
    reid_images_dir = "images"
    reid_masks_dir = "masks"
    reid_kps_dir = "keypoints"
    reid_fig_dir = "figures"
    reid_anns_dir = "anns"
    images_anns_filename = "reid_crops_anns.json"
    masks_anns_filename = "reid_masks_anns.json"
    kps_anns_filename = "reid_kps_anns.json"
    dataset_sampling_filename = "dataset_sampling.json"
    train_dir = 'gaussian_joints'
    dataset_dir = ""

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        "keypoints": (17, False, ".npy", ["p{}".format(p) for p in range(1, 17)],),
        "keypoints_gaussian": (17, False, ".npy", ["p{}".format(p) for p in range(1, 17)],),
        "joints": (10, False, ".npy", ["p{}".format(p) for p in range(1, 17)]),
        "joints_gaussian": (10, False, ".npy", ["p{}".format(p) for p in range(1, 17)]),
        "pose_on_img": (35, False, ".npy", ["p{}".format(p) for p in range(1, 35)]),
        "pose_on_img_crops": (35, False, ".npy", ["p{}".format(p) for p in range(1, 35)]),
    }

    pifpaf_config = CN()
    pifpaf_config.predict = CN()
    pifpaf_config.predict["checkpoint"] = "shufflenetv2k30"
    pifpaf_config.predict["long-edge"] = 256
    pifpaf_config.predict["quiet"] = None
    pifpaf_config.predict["dense-connections"] = None
    pifpaf_config.predict["seed-threshold"] = 0.2
    pifpaf_config.predict["instance-threshold"] = 0.15
    pifpaf_config.predict["decoder-workers"] = 8

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in ReidDataset.masks_dirs:
            return None
        else:
            return ReidDataset.masks_dirs[masks_dir]

    def gallery_filter(self, q_pid, q_camid, q_ann, g_pids, g_camids, g_anns):
        """camid refers to video id: remove gallery samples from the different videos than query sample"""
        if self.eval_metric == 'mot_inter_intra_video':
            return np.array(np.zeros_like(g_pids), dtype=bool)
        elif self.eval_metric == 'mot_inter_video':
            remove = g_camids == q_camid
            return remove
        elif self.eval_metric == 'mot_intra_video':
            remove = g_camids != q_camid
            return remove
        else:
            raise ValueError

    def __init__(
        self,
        reid_config,
        tracking_dataset: TrackingDataset,
        masks_dir="",
        root="",
        occluded_dataset=False,  # sample most occluded images as queries in the test set
        config=None,
        **kwargs
    ):
        self.tracking_dataset = tracking_dataset
        self.reid_config = reid_config
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # Init
        # self.pose_model = None
        self.pose_model = OpenPifPaf(self.pifpaf_config, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=16)

        self.pose_dl = None
        self.pose_datapipe = None
        self.dataset_path = Path(self.tracking_dataset.dataset_path)
        self.masks_dir = masks_dir
        self.column_mapping = {}

        self.eval_metric = self.reid_config.eval_metric if tracking_dataset.name != 'posetrack21' else 'mot_inter_intra_video'
        self.multi_video_queries_only = self.reid_config.multi_video_queries_only

        val_set = self.tracking_dataset.sets[self.reid_config.test.set_name]
        train_set = self.tracking_dataset.sets[self.reid_config.train.set_name]
        if self.reid_config.train.set_name in self.tracking_dataset.set_split_idxs:
            set_split_idx = self.tracking_dataset.set_split_idxs[self.reid_config.train.set_name]
            self.reid_dir = self.reid_dir + "_" + str(set_split_idx)

        self.occluded_dataset = occluded_dataset
        self.sam_checkpoint = osp.abspath(osp.expanduser(self.reid_config.sam_checkpoint))
        self.enable_sam = self.reid_config.enable_sam
        self.enable_dataset_sampling_loading = False

        assert (
            self.reid_config.train.max_samples_per_id
            >= self.reid_config.train.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"
        assert (
            self.reid_config.test.max_samples_per_id
            >= self.reid_config.test.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"

        if self.masks_dir in self.masks_dirs:
            (
                self.masks_parts_numbers,
                self.has_background,
                self.masks_suffix,
                self.masks_parts_names,
            ) = self.masks_dirs[self.masks_dir]
        else:
            (
                self.masks_parts_numbers,
                self.has_background,
                self.masks_suffix,
                self.masks_parts_names,
            ) = (None, None, None, None)

        # Build ReID dataset from MOT dataset
        self.build_reid_set(
            train_set,
            self.reid_config,
            "train",
            is_test_set=False,
        )

        self.build_reid_set(
            val_set,
            self.reid_config,
            "val",
            is_test_set=True,
        )

        self.train_gt_dets = train_set.detections_gt
        self.val_gt_dets = val_set.detections_gt

        # Get train/query/gallery sets as torchreid list format
        self.train_df = self.train_gt_dets[self.train_gt_dets["split"] == "train"]
        self.query_df = self.val_gt_dets[self.val_gt_dets["split"] == "query"]
        self.gallery_df = self.val_gt_dets[self.val_gt_dets["split"] == "gallery"]
        assert len(self.train_df) > 0, "An error occurred, no train samples found"
        assert len(self.query_df) > 0, "An error occurred, no query samples found"
        assert len(self.gallery_df) > 0, "An error occurred, no gallery samples found"

        train, query, gallery = self.to_torchreid_dataset_format(
            [self.train_df, self.query_df, self.gallery_df]
        )

        super().__init__(train, query, gallery, config=config, **kwargs)

        self.name = tracking_dataset.name

    def build_reid_set(self, tracking_set, reid_config, split, is_test_set):
        """
        Build ReID metadata for a given MOT dataset split.
        Only a subset of all MOT groundtruth detections is used for ReID.
        Detections to be used for ReID are selected according to the filtering criteria specified in the config 'reid_cfg'.
        If "enable_dataset_sampling_loading" is set, the sampling annotations are loaded from disk to assign each
        detection a "split" value, that can be "train"/"none" for the train set and "query"/"gallery"/"none" for the test
         set (ReID test set = tracking validation set).
        Image crops and human parsing labels (masks) are generated for each selected detection only.
        If the config is changed and more detections are selected, the image crops and masks are generated only for
        these new detections.
        """
        image_metadatas = tracking_set.image_metadatas
        detections = tracking_set.detections_gt
        fig_size = reid_config.fig_size
        mask_size = reid_config.mask_size
        max_crop_size = reid_config.max_crop_size
        reid_set_cfg = reid_config.test if is_test_set else reid_config.train
        masks_mode = reid_config.masks_mode

        log.info("Loading {} set...".format(split))

        # Precompute all paths
        reid_path = Path(self.dataset_path, self.reid_dir) if self.reid_config.enable_human_parsing_labels else Path(self.dataset_path, self.reid_dir)
        reid_img_path = reid_path / self.reid_images_dir / split
        reid_mask_path = reid_path / self.reid_masks_dir / split
        reid_kp_path = reid_path / self.reid_kps_dir / split
        reid_fig_path = reid_path / self.reid_fig_dir / split
        reid_anns_filepath = (
            reid_path
            / self.reid_images_dir
            / self.reid_anns_dir
            / (split + "_" + self.images_anns_filename)
        )
        masks_anns_filepath = (
            reid_path
            / self.reid_masks_dir
            / self.reid_anns_dir
            / (split + "_" + self.masks_anns_filename)
        )
        kps_anns_filepath = (
            reid_path
            / self.reid_kps_dir
            / self.reid_anns_dir
            / (split + "_" + self.kps_anns_filename)
        )
        dataset_sampling_path = Path(self.dataset_path, self.reid_dir) / (split + "_" + self.dataset_sampling_filename)

        # Load reid crops metadata into existing ground truth detections dataframe
        self.load_reid_annotations(
            detections,
            reid_anns_filepath,
            ["reid_crop_path", "reid_crop_width", "reid_crop_height", "negative_kps"],
        )

        if "negative_kps" in detections.columns:
            # Add negative keypoints to each detection
            detections["negative_kps"] = detections["negative_kps"].apply(lambda x: np.array(x) if (isinstance(x, list) and len(x) > 0) else np.empty((0, 17, 3)))

        # Load reid masks metadata into existing ground truth detections dataframe
        self.load_reid_annotations(detections, masks_anns_filepath, ["masks_path"])
        if not "keypoints_xyc" in detections.columns:
            self.load_reid_annotations(detections, kps_anns_filepath, ["kp_path"])

        # Sampling of detections to be used to create the ReID dataset
        if self.enable_dataset_sampling_loading:
            self.load_dataset_sampling(detections, dataset_sampling_path)
        else:
            self.sample_detections_for_reid(detections, reid_set_cfg)

        # Save ReID detections crops and related metadata. Apply only on sampled detections
        self.save_reid_img_crops(
            detections,
            reid_img_path,
            split,
            reid_anns_filepath,
            image_metadatas,
            max_crop_size,
        )

        # Save human parsing pseudo ground truth and related metadata. Apply only on sampled detections
        if self.reid_config.enable_human_parsing_labels:
            self.save_reid_masks_crops(
                detections,
                reid_img_path,
                reid_mask_path,
                reid_kp_path,
                reid_fig_path,
                split,
                masks_anns_filepath,
                kps_anns_filepath,
                image_metadatas,
                fig_size,
                mask_size,
                mode=masks_mode,
            )
        else:
            detections["masks_path"] = ''
            if not "keypoints_xyc" in detections.columns:
                detections["kp_path"] = ''

        # Add 0-based pid column (for Torchreid compatibility) to sampled detections
        self.add_pid_column(detections)
        if "keypoints_xyc" in detections.columns:
            self.add_occlusion_level_column(detections)

        # Flag sampled detection as a query or gallery if this is a test set
        if is_test_set:
            self.query_gallery_split(detections, reid_set_cfg.ratio_query_per_id)

        # Save selected detections metadata to disk
        # self.save_dataset_sampling(detections, dataset_sampling_path)

        # Turn path into absolute path
        detections['masks_path'] = detections['masks_path'].apply(lambda x: str(reid_mask_path / x) if x and x is not np.nan else None)
        if not "keypoints_xyc" in detections.columns:
            detections['kp_path'] = detections['kp_path'].apply(lambda x: str(reid_kp_path / x) if x and x is not np.nan else None)
        detections['reid_crop_path'] = detections['reid_crop_path'].apply(lambda x: str(reid_img_path / x) if x and x is not np.nan else None)

    def save_dataset_sampling(self, detections, dataset_sampling_path):
        log.info(
            'Saving dataset sampling annotations as json to "{}"'.format(dataset_sampling_path)
        )
        dataset_sampling_path.parent.mkdir(parents=True, exist_ok=True)
        detections[
            ["id", "split"]
        ].to_json(dataset_sampling_path)

    def add_negative_samples(self, _df):
        all_kps_in_img = np.array(list(_df.keypoints_xyc))
        id_to_index = {k: v for v, k in enumerate(list(_df.id))}
        _df["negative_kps"] = _df\
            .apply(lambda bb: keypoints_in_bbox_coord(np.delete(all_kps_in_img, id_to_index[bb.id], axis=0), bb.bbox_ltwh), axis=1)\
            .apply(lambda kp_xyc_bbox: kp_xyc_bbox[kp_xyc_bbox[:, :, 2].sum(axis=1) > 0]) # remove non visibile skeletons

        return _df

    def load_reid_annotations(self, gt_dets, reid_anns_filepath, columns):
        if reid_anns_filepath.exists():
            reid_anns = pd.read_json(
                reid_anns_filepath, convert_dates=False, convert_axes=False
            )
            reid_anns.set_index("id", drop=False, inplace=True)
            tmp_df = gt_dets.merge(
                reid_anns,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )
            common_columns = [col for col in columns if col in tmp_df.columns]
            gt_dets[common_columns] = tmp_df[common_columns]
        else:
            # no annotations yet, initialize empty columns
            for col in columns:
                gt_dets[col] = None

    def load_dataset_sampling(self, dets_df, dataset_sampling_path):
        if dataset_sampling_path.exists():
            sampling_anns = pd.read_json(
                dataset_sampling_path, convert_dates=False, convert_axes=False
            )
            sampling_anns.set_index("id", drop=False, inplace=True)

            # Drop the 'split' column since it should be overwritten by the sampling file
            if "split" in dets_df.columns:
                dets_df.drop(columns=['split'], inplace=True)

            tmp_df = dets_df.merge(
                sampling_anns,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )
            dets_df["split"] = tmp_df["split"]
        else:
            raise FileNotFoundError("Dataset sampling file not found ({}). Please follow the instructions on the main repository to download the file and place it under this location.".format(dataset_sampling_path))

    def sample_detections_for_reid(self, dets_df, reid_cfg):
        dets_df["split"] = "none"

        # Filter detections by visibility
        dets_df_f1 = dets_df[dets_df.visibility >= reid_cfg.min_vis]

        # Filter detections by crop size
        keep = dets_df_f1.bbox_ltwh.apply(
            lambda x: x[2] > reid_cfg.min_w
        ) & dets_df_f1.bbox_ltwh.apply(lambda x: x[3] > reid_cfg.min_h)
        dets_df_f2 = dets_df_f1[keep]
        log.warning(
            "{} removed because too small samples (h<{} or w<{}) = {}".format(
                self.__class__.__name__,
                (reid_cfg.min_h),
                (reid_cfg.min_w),
                len(dets_df_f1) - len(dets_df_f2),
            )
        )

        # Filter detections by uniform sampling along each tracklet
        dets_df_f3 = (
            dets_df_f2.groupby("person_id")
            .apply(
                self.uniform_tracklet_sampling, reid_cfg.max_samples_per_id, "image_id"
            )
            .reset_index(drop=True)
            .copy()
        )
        log.warning(
            "{} removed for uniform tracklet sampling = {}".format(
                self.__class__.__name__, len(dets_df_f2) - len(dets_df_f3)
            )
        )

        # Keep only ids with at least MIN_SAMPLES appearances
        count_per_id = dets_df_f3.person_id.value_counts()
        ids_to_keep = count_per_id.index[count_per_id.ge((reid_cfg.min_samples_per_id))]
        dets_df_f4 = dets_df_f3[dets_df_f3.person_id.isin(ids_to_keep)]
        log.warning(
            "{} removed for not enough samples per id = {}".format(
                self.__class__.__name__, len(dets_df_f3) - len(dets_df_f4)
            )
        )

        # Keep only max_total_ids ids
        if reid_cfg.max_total_ids == -1 or reid_cfg.max_total_ids > len(
            dets_df_f4.person_id.unique()
        ):
            reid_cfg.max_total_ids = len(dets_df_f4.person_id.unique())
        # reset seed to make sure the same split is used if the dataset is instantiated multiple times
        np.random.seed(0)
        ids_to_keep = np.random.choice(
            dets_df_f4.person_id.unique(), replace=False, size=reid_cfg.max_total_ids
        )
        dets_df_f5 = dets_df_f4[dets_df_f4.person_id.isin(ids_to_keep)]

        dets_df.loc[dets_df.id.isin(dets_df_f5.id), "split"] = "train"
        log.info(
            "{} filtered size = {}".format(self.__class__.__name__, len(dets_df_f5))
        )

    def save_reid_img_crops(
        self,
        gt_dets,
        save_path,
        set_name,
        reid_anns_filepath,
        metadatas_df,
        max_crop_size,
    ):
        """
        Save on disk all detections image crops from the ground truth dataset to build the reid dataset.
        Create a json annotation file with crops metadata.
        """
        save_path = save_path
        max_h, max_w = max_crop_size
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.reid_crop_path.isnull()
        ]
        if len(gt_dets_for_reid) == 0:
            log.info(
                "All detections used for ReID already have their image crop saved on disk."
            )
            return

        # compute negative keypoints to be saved on disk
        if "keypoints_xyc" in gt_dets.columns:
            gt_dets["negative_kps"] = gt_dets.groupby("image_id").apply(self.add_negative_samples).reset_index(level=0, drop=True)["negative_kps"]
            gt_dets_for_reid = gt_dets[
                (gt_dets.split != "none") & gt_dets.reid_crop_path.isnull()
            ]
        # gt_dets_for_reid.reset_index(drop=True, inplace=True)
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} reid crops".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                video_id = str(video_id)
                img_metadata = metadatas_df[metadatas_df.id == image_id].iloc[0]
                img = cv2.imread(img_metadata.file_path)
                for det_metadata in dets_from_img.itertuples():
                    # crop and resize bbox from image
                    bbox_ltwh = det_metadata.bbox_ltwh
                    bbox_ltwh = clip_bbox_ltwh_to_img_dim(
                        bbox_ltwh, img.shape[1], img.shape[0]
                    )
                    pid = det_metadata.person_id
                    l, t, w, h = bbox_ltwh.astype(int)
                    img_crop = img[t : t + h, l : l + w]
                    if h > max_h or w > max_w:
                        img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)

                    # save crop to disk
                    filename = "{}_{}_{}{}".format(
                        pid, video_id, img_metadata.id, self.img_ext
                    )
                    rel_filepath = Path(str(video_id), filename)
                    abs_filepath = Path(save_path, rel_filepath)
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(abs_filepath), img_crop)

                    # save image crop metadata
                    gt_dets.at[det_metadata.Index, "reid_crop_path"] = str(rel_filepath)
                    gt_dets.at[det_metadata.Index, "reid_crop_width"] = img_crop.shape[1]
                    gt_dets.at[det_metadata.Index, "reid_crop_height"] = img_crop.shape[0]
                    pbar.update(1)

        log.info(
            'Saving reid crops annotations as json to "{}"'.format(reid_anns_filepath)
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        columns = ["id", "reid_crop_path", "reid_crop_width", "reid_crop_height"]
        if "negative_kps" in gt_dets.columns:
            columns.append("negative_kps")
        gt_dets[
            columns
        ].to_json(reid_anns_filepath)

    def save_reid_masks_crops(
        self,
        gt_dets,
        reid_img_path,
        masks_save_path,
        kps_save_path,
        fig_save_path,
        set_name,
        masks_anns_filepath,
        kps_anns_filepath,
        metadatas_df,
        fig_size,
        masks_size,
        mode="keypoints_gaussian",
    ):
        """
        Save on disk all human parsing gt for each reid crop.
        Create a json annotation file with human parsing metadata.
        """
        fig_h, fig_w = fig_size
        mask_h, mask_w = masks_size
        g_scale = 10
        g_radius = int(mask_w / g_scale)
        gaussian = gkern(g_radius * 2 + 1)
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.masks_path.isnull()
        ]
        if mode == "none":
            log.info("No human parsing labels to compute for this mode.")
            return
        if len(gt_dets_for_reid) == 0:
            log.info("All reid crops already have human parsing masks labels.")
            return
        if (mode == "pose_on_img_crops" or mode == "pose_on_img") and self.enable_sam:
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
            sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
            predictor = SamPredictor(sam)
        kp_grouping_eight_bp = CocoToEightBodyMasks()
        # kp_grouping_eight_bp = None
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} human parsing labels".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                video_id = str(video_id)
                img_metadata = metadatas_df[metadatas_df.id == image_id].iloc[0]
                # load image once to get video frame size
                if mode == "pose_on_img":
                    if self.pose_dl == None:  # TODO
                        self.pose_dl = DataLoader(
                            dataset=self.pose_datapipe,
                            batch_size=128,
                            num_workers=0,
                            collate_fn=type(self.pose_model).collate_fn,
                            persistent_workers=False,
                        )
                    fields_list = []
                    self.pose_datapipe.update(
                        metadatas_df[metadatas_df.id == image_id], None
                    )
                    for idxs, pose_batch in self.pose_dl:
                        batch_metadatas = metadatas_df.loc[idxs]
                        _, fields = self.pose_model.process(
                            pose_batch, batch_metadatas, return_fields=True
                        )
                        fields_list.extend(fields)

                    masks_gt_or = torch.concat(
                        (
                            fields_list[0][0][:, 1],
                            fields_list[0][1][:, 1],
                        )
                    )
                    img = cv2.imread(img_metadata.file_path)
                    masks_gt = resize(
                        masks_gt_or.numpy(),
                        (masks_gt_or.numpy().shape[0], img.shape[0], img.shape[1]),
                    )

                # loop on detections in frame
                for det_metadata in dets_from_img.itertuples():
                    img_crop_or = cv2.imread(str(Path(reid_img_path, det_metadata.reid_crop_path)))
                    l, t, w, h = det_metadata.bbox_ltwh
                    img_crop = cv2.resize(img_crop_or, (fig_w, fig_h), cv2.INTER_CUBIC)
                    keypoints_xyc = None
                    negative_kps_xyc_crop = None
                    if "keypoints_xyc" in gt_dets.columns:
                        kps_xyc_or = kp_img_to_kp_bbox(det_metadata.keypoints_xyc, det_metadata.bbox_ltwh)
                        keypoints_xyc = rescale_keypoints(
                            kps_xyc_or,
                            (w, h),
                            (mask_w, mask_h),
                        )
                        assert ((keypoints_xyc[:, 0] >= 0) & (keypoints_xyc[:, 0] < mask_w)).all()
                        assert ((keypoints_xyc[:, 1] >= 0) & (keypoints_xyc[:, 1] < mask_h)).all()
    
                        keypoints_xyc_crop = clip_keypoints_to_image(kps_xyc_or, (w, h))
                        keypoints_xyc_crop = rescale_keypoints(keypoints_xyc_crop, (w, h), (fig_w, fig_h))

                        negative_kps_xyc = det_metadata.negative_kps
                        negative_kps_xyc = clip_keypoints_to_image(negative_kps_xyc, (w, h))
                        negative_kps_xyc_crop = rescale_keypoints(negative_kps_xyc, (w, h), (fig_w, fig_h))

                    if mode == "keypoints":
                        # compute human parsing heatmaps as gaussian on each visible keypoint
                        masks_gt_crop = build_keypoints_heatmaps(
                            keypoints_xyc, mask_w, mask_h
                        )
                    elif mode == "keypoints_gaussian":
                        # compute human parsing heatmaps as gaussian on each visible keypoint
                        masks_gt_crop = build_keypoints_gaussian_heatmaps(
                            keypoints_xyc, mask_w, mask_h, gaussian=gaussian
                        )
                    elif mode == "joints":
                        # compute human parsing heatmaps as shapes around on each visible keypoint
                        masks_gt_crop = build_joints_heatmaps(
                            keypoints_xyc, mask_w, mask_h
                        )
                    elif mode == "joints_gaussian":
                        # compute human parsing heatmaps as shapes around on each visible keypoint
                        masks_gt_crop = build_joints_gaussian_heatmaps(
                            keypoints_xyc, mask_w, mask_h
                        )
                    elif mode == "pose_on_img_crops":
                        # compute human parsing heatmaps using output of pose model on cropped person image
                        pim_img_crop = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
                        processed_image, anns, meta = self.pose_model.pifpaf_preprocess(pim_img_crop, [], {})
                        processed_image = processed_image.unsqueeze(0)
                        results, fields_batch = self.pose_model.processor.batch(
                            self.pose_model.model, processed_image, device=self.pose_model.device
                        )
                        if "keypoints_xyc" not in gt_dets.columns:
                            # Use pifpaf keypoints if not provided with ground truth annotations
                            skeletons = [annotation.data for annotation in results[0]]
                            pifpaf_h, pifpaf_w = processed_image.shape[-2:]
                            keypoints_xyc_pifpaf, negative_kps_xyc_pifpaf = compute_target_skeleton(skeletons, pifpaf_w, pifpaf_h)

                            keypoints_xyc_pifpaf = clip_keypoints_to_image(keypoints_xyc_pifpaf, (pifpaf_w, pifpaf_h))
                            keypoints_xyc_crop = rescale_keypoints(keypoints_xyc_pifpaf, (pifpaf_w, pifpaf_h), (fig_w, fig_h))
                            keypoints_xyc = rescale_keypoints(keypoints_xyc_pifpaf, (pifpaf_w, pifpaf_h), (img_crop_or.shape[1], img_crop_or.shape[0]))

                            negative_kps_xyc_pifpaf = clip_keypoints_to_image(negative_kps_xyc_pifpaf, (pifpaf_w, pifpaf_h))
                            negative_kps_xyc_crop = rescale_keypoints(negative_kps_xyc_pifpaf, (pifpaf_w, pifpaf_h), (fig_w, fig_h))
                            negative_kps_xyc = rescale_keypoints(negative_kps_xyc_pifpaf, (pifpaf_w, pifpaf_h), (img_crop_or.shape[1], img_crop_or.shape[0]))

                        masks_gt_crop = torch.concat(
                            (
                                fields_batch[0][0][:, 1],
                                fields_batch[0][1][:, 1],
                            )
                        )
                        masks_gt_crop = masks_gt_crop.unsqueeze(0)

                        masks_gt_crop = F.interpolate(
                            masks_gt_crop,
                            size=(mask_h, mask_w),
                            mode="bilinear",
                            align_corners=True
                        )

                        masks_gt_crop = masks_gt_crop.squeeze().numpy()
                        kernel = np.ones((10, 10), np.uint8)
                        if self.enable_sam and keypoints_xyc_crop is not None and len(keypoints_xyc_crop) > 0:
                            # pifpaf body part masks are too coarse (overlap background) and cover all humans in
                            # the bbox. Compute a SAM segmentation mask with the pifpaf keypoints of the target person
                            # as prompt, and only keep pif and paf field inside that SAM ask.
                            sam_mask = self.compute_sam_mask(predictor, img_crop, keypoints_xyc_crop, negative_kps_xyc_crop)
                            sam_mask = cv2.dilate(sam_mask.astype(np.uint8), kernel, iterations=2)

                            sam_mask = cv2.resize(sam_mask.squeeze(), (mask_w, mask_h))
                            #
                            masks_gt_crop = masks_gt_crop * sam_mask
                    elif mode == "pose_on_img":
                        # compute human parsing heatmaps using output of pose model on full image
                        bbox_ltwh = clip_bbox_ltwh_to_img_dim(
                            det_metadata.bbox_ltwh, img.shape[1], img.shape[0]
                        ).astype(int)
                        l, t, w, h = bbox_ltwh
                        img_crop = img[t : t + h, l : l + w]
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        masks_gt_crop = masks_gt[:, t : t + h, l : l + w]
                        masks_gt_crop = resize(
                            masks_gt_crop, (masks_gt_crop.shape[0], fig_h, fig_w)
                        )
                        sam_mask = self.compute_sam_mask(predictor, img_crop, keypoints_xyc_crop, negative_kps_xyc_crop)
                        masks_gt_crop = masks_gt_crop * sam_mask
                    else:
                        raise ValueError("Invalid human parsing method '{}'".format(mode))

                    # save human parsing heatmaps on disk
                    pid = det_metadata.person_id
                    filename = "{}_{}_{}".format(pid, video_id, image_id)
                    rel_masks_filepath = Path(video_id, filename + self.masks_ext)
                    abs_masks_filepath = Path(
                        masks_save_path, rel_masks_filepath
                    )
                    abs_masks_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(abs_masks_filepath), masks_gt_crop)

                    # save skeletons on disk
                    if not "keypoints_xyc" in gt_dets.columns:
                        rel_kps_filepath = Path(video_id, filename + self.kps_ext)
                        abs_kps_filepath = Path(
                            kps_save_path, rel_kps_filepath
                        )
                        abs_kps_filepath.parent.mkdir(parents=True, exist_ok=True)
                        skeletons_json = create_skeletons_json(keypoints_xyc, negative_kps_xyc)
                        with open(abs_kps_filepath, "w") as fp:
                            json.dump(skeletons_json, fp)

                    # save image crop with human parsing heatmaps overlayed on disk for visualization/debug purpose
                    img_with_heatmap = colored_body_parts_overlay(
                        img_crop, masks_gt_crop
                    )
                    figure_filepath = Path(
                        fig_save_path, video_id, filename + "_heatmaps_" + self.img_ext
                    )
                    figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(figure_filepath), img_with_heatmap)
                    img_crop_kps = img_crop
                    if keypoints_xyc_crop is not None and len(keypoints_xyc_crop) > 0:
                        keypoints_xyck_crop = kp_grouping_eight_bp.apply_to_keypoints_xyc(keypoints_xyc_crop)
                        img_crop_kps = draw_keypoints(img_crop, keypoints_xyck_crop, (fig_w, fig_h), radius=2, thickness=2)
                    if negative_kps_xyc_crop is not None and len(negative_kps_xyc_crop) > 0:
                        for negative_kps in negative_kps_xyc_crop:
                            negative_kps_xyck = kp_grouping_eight_bp.apply_to_keypoints_xyc(negative_kps)
                            img_crop_kps = draw_keypoints(img_crop_kps, negative_kps_xyck, (fig_w, fig_h), radius=2, thickness=2, color=(0, 0, 255))
                    kps_img_filepath = Path(
                        fig_save_path, video_id, filename + "_kps_"  + self.img_ext
                    )
                    cv2.imwrite(str(kps_img_filepath), img_crop_kps)
                    # record human parsing metadata for later json dump
                    gt_dets.at[det_metadata.Index, "masks_path"] = str(rel_masks_filepath)
                    if not "keypoints_xyc" in gt_dets.columns:
                        gt_dets.at[det_metadata.Index, "kp_path"] = str(rel_kps_filepath)
                    pbar.update(1)
        log.info(
            'Saving reid human parsing annotations as json to "{}"'.format(
                masks_anns_filepath
            )
        )
        masks_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets[["id", "masks_path"]].to_json(masks_anns_filepath)
        if not "keypoints_xyc" in gt_dets.columns:
            kps_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
            gt_dets[["id", "kp_path"]].to_json(kps_anns_filepath)

    def compute_sam_mask(self, predictor, img_crop, keypoints_xyc_crop, neg_kps_xyc):
        predictor.set_image(img_crop, image_format="BGR")
        keypoints_xyc_crop = keypoints_xyc_crop[keypoints_xyc_crop[:, -1] > 0]
        neg_kps_xyc = neg_kps_xyc.reshape((-1, 3))
        neg_kps_xyc = neg_kps_xyc[neg_kps_xyc[:, -1] > 0]
        all_keypoints = np.concatenate((keypoints_xyc_crop, neg_kps_xyc))
        keypoints_labels = np.array([1] * len(keypoints_xyc_crop) + [0] * len(neg_kps_xyc))
        sam_mask, _, _ = predictor.predict(point_coords=all_keypoints[:, :2], point_labels=keypoints_labels,
                                           multimask_output=False)
        return sam_mask

    def rescale_and_filter_keypoints(self, keypoints, bbox_ltwh, new_w, new_h):
        l, t, w, h = bbox_ltwh.astype(int)
        discarded_keypoints = 0
        rescaled_keypoints = {}
        for i, kp in enumerate(keypoints):
            # remove unvisible keypoints
            if kp[2] == 0:
                continue

            # put keypoints in bbox coord space
            kpx, kpy = kp[:2].astype(int) - np.array([l, t])

            # remove keypoints out of bbox
            if kpx < 0 or kpx >= w or kpy < 0 or kpy >= h:
                discarded_keypoints += 1
                continue

            # put keypoints in resized image coord space
            kpx, kpy = kpx * new_w / w, kpy * new_h / h

            rescaled_keypoints[i] = np.array([int(kpx), int(kpy), 1])
        return rescaled_keypoints, discarded_keypoints

    def query_gallery_split(self, gt_dets, ratio):
        def random_tracklet_sampling(_df):
            x = list(_df.index)
            size = ceil(len(x) * ratio)
            result = list(np.random.choice(x, size=size, replace=False))
            return _df.loc[result]

        def occlusion_tracklet_sampling(_df):
            _df = _df.sort_values(by=['occ_level'], ascending=False)
            indices = list(_df.index)
            result = indices[:int(len(indices) * ratio)]
            return _df.loc[result]

        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        # reset seed to make sure the same split is used if the dataset is instantiated multiple times
        np.random.seed(0)
        sampling = occlusion_tracklet_sampling if self.occluded_dataset else random_tracklet_sampling
        queries_per_pid = gt_dets_for_reid.groupby("person_id").apply(
            sampling
        )
        if self.eval_metric == 'mot_inter_video' or self.multi_video_queries_only:
            # keep only queries that are in more than one video
            queries_per_pid = queries_per_pid.droplevel(level=0).groupby("person_id")['video_id'].filter(lambda g: (g.nunique() > 1)).reset_index()
            assert len(queries_per_pid) != 0, "There were no identity with more than one videos to be used as queries. " \
                                              "Try setting 'multi_video_queries_only' to False or not using " \
                                              "eval_metric='mot_inter_video' or adjust the settings to sample a " \
                                              "bigger ReID dataset."
        gt_dets.loc[gt_dets.split != "none", "split"] = "gallery"
        gt_dets.loc[gt_dets.id.isin(queries_per_pid.id), "split"] = "query"

    def to_torchreid_dataset_format(self, dataframes):
        results = []
        for df in dataframes:
            df = df.copy()  # to avoid SettingWithCopyWarning
            # use video id as camera id: camid is used at inference to filter out gallery samples given a query sample
            df["camid"] = pd.Categorical(df.video_id, categories=df.video_id.unique()).codes
            df["videoid"] = pd.Categorical(df.video_id, categories=df.video_id.unique()).codes
            df["img_path"] = df["reid_crop_path"]
            if "keypoints_xyc" in df.columns:
                df["keypoints_xyc"] = df.apply(lambda r: kp_img_to_kp_bbox(r.keypoints_xyc, r.bbox_ltwh), axis=1)
                df["keypoints_xyc"] = df.apply(lambda r: rescale_keypoints(r.keypoints_xyc, (r.bbox_ltwh[2], r.bbox_ltwh[3]), (r.reid_crop_width, r.reid_crop_height)), axis=1)
                df["negative_kps"] = df.apply(lambda r: rescale_keypoints(r.negative_kps, (r.bbox_ltwh[2], r.bbox_ltwh[3]), (r.reid_crop_width, r.reid_crop_height)), axis=1)

            # remove bbox_head as it is not available for each sample
            # df to list of dict
            sorted_df = df.sort_values(by=["pid"])
            # use only necessary annotations: using them all caused a
            # 'RuntimeError: torch.cat(): input types can't be cast to the desired output type Long' in collate.py
            # -> still has to be fixed
            sorted_df['datasetid'] = 'occluded_posetrack'
            filter_cols = ["pid", "camid", "video_id", "img_path", "masks_path", "kp_path", "visibility", "keypoints_xyc", "reid_crop_width", "reid_crop_height", "negative_kps", "occ_level", "datasetid", "videoid"]
            for col in filter_cols:
                if col not in sorted_df.columns:
                    filter_cols.remove(col)
            data_list = sorted_df[filter_cols]
            data_list = data_list.to_dict("records")
            results.append(data_list)
        return results

    def add_pid_column(self, gt_dets):
        # create pids as 0-based increasing numbers
        gt_dets["pid"] = None
        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        gt_dets.loc[gt_dets_for_reid.index, "pid"] = pd.factorize(
            gt_dets_for_reid.person_id
        )[0]

    def add_occlusion_level_column(self, gt_dets):
        def compute_occlusion_score(r):
            if r.keypoints_xyc[..., 2].sum() == 0:
                return r.negative_kps[..., 2].sum() * 2
            return r.negative_kps[..., 2].sum() / r.keypoints_xyc[..., 2].sum()
        gt_dets["occ_level"] = gt_dets.apply(compute_occlusion_score, axis=1)

    def uniform_tracklet_sampling(self, _df, max_samples_per_id, column):
        _df.sort_values(column)
        num_det = len(_df)
        if num_det > max_samples_per_id:
            # Select 'max_samples_per_id' evenly spaced indices, including first and last
            indices = np.round(np.linspace(0, num_det - 1, max_samples_per_id)).astype(
                int
            )
            assert len(indices) == max_samples_per_id
            return _df.iloc[indices]
        else:
            return _df


def compute_target_skeleton(skeletons, W, H):
    """
    Computes the target skeleton and the remaining skeletons from a list of skeletons.

    Parameters:
        skeletons (list of np.ndarray): List of skeletons, each of size [17, 3].
        W (int): Width of the image crop.
        H (int): Height of the image crop.

    Returns:
        keypoints_xyc (np.ndarray): The target skeleton of size [17, 3], or an empty array if no skeletons.
        negative_kps_xyc (np.ndarray): The remaining skeletons of size [N, 17, 3], or an empty array if no remaining skeletons.
    """
    # Handle cases with no skeletons
    if not skeletons:
        return np.empty((0, 17, 3)), np.empty((0, 17, 3))

    # If there's only one skeleton, use it directly as the target
    if len(skeletons) == 1:
        return skeletons[0], np.empty((0, 17, 3))

    # Define the top center region
    top_center_x = W / 2
    top_threshold_y = H * 0.2

    # Initialize variables to track the closest skeleton
    closest_skeleton = None
    min_distance = float('inf')
    target_index = -1

    for i, skeleton in enumerate(skeletons):
        head_keypoint = skeleton[0]  # Assuming keypoint 0 is the head
        head_x, head_y, confidence = head_keypoint

        # Compute distance to the top center part of the image
        if confidence > 0:  # Only consider keypoints with some confidence
            distance = np.sqrt((head_x - top_center_x) ** 2 + head_y ** 2)
            if head_y < top_threshold_y and distance < min_distance:
                min_distance = distance
                closest_skeleton = skeleton
                target_index = i

    # If no valid skeleton is found within the region, use the first skeleton as the target
    if closest_skeleton is None:
        closest_skeleton = skeletons[0]
        target_index = 0

    # Prepare the target skeleton
    keypoints_xyc = closest_skeleton

    # Construct the negative_kps_xyc array with the remaining skeletons
    negative_kps_xyc = np.array([skeleton for i, skeleton in enumerate(skeletons) if i != target_index])

    # Ensure negative_kps_xyc has the correct shape, even if empty
    if negative_kps_xyc.size == 0:
        negative_kps_xyc = np.empty((0, 17, 3))

    return keypoints_xyc, negative_kps_xyc



def create_skeletons_json(keypoints_xyc, negative_kps_xyc):
    """
    Combines keypoints_xyc and negative_kps_xyc into a single list of skeletons with an 'is_target' attribute.

    Parameters:
        keypoints_xyc (np.ndarray): Target skeleton keypoints of size [17, 3] or an empty array.
        negative_kps_xyc (np.ndarray): Remaining skeletons keypoints of size [N, 17, 3] or an empty array.

    Returns:
        list: A list of dictionaries representing skeletons with keypoints and 'is_target' attributes.
    """
    skeletons_list = []

    # Add the target skeletons
    if keypoints_xyc.size > 0:
        skeletons_list.append({
            "keypoints": keypoints_xyc.tolist(),
            "bbox": [],  # Assuming bbox can be filled later or added if available
            "score": 0,  # Placeholder for score, can be updated
            "category_id": 1,  # Assuming a default category_id, can be updated
            "is_target": True
        })

    # Add the remaining skeletons
    if negative_kps_xyc.size > 0:
        for skeleton in negative_kps_xyc:
            skeletons_list.append({
                "keypoints": skeleton.tolist(),
                "bbox": [],  # Assuming bbox can be filled later or added if available
                "score": 0,  # Placeholder for score, can be updated
                "category_id": 1,  # Assuming a default category_id, can be updated
                "is_target": False
            })

    return skeletons_list
