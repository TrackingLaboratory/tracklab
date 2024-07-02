from __future__ import absolute_import, division, print_function

import sys
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from math import ceil
from pathlib import Path
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracklab.datastruct import EngineDatapipe, TrackingDataset
from tracklab.utils.coordinates import (
    rescale_keypoints,
    clip_keypoints_to_image, keypoints_in_bbox_coord,
)
from tracklab.utils.cv2 import colored_body_parts_overlay
import tracklab

from kpreid.data import ImageDataset
from kpreid.utils.imagetools import (
    gkern, build_joints_heatmaps, build_joints_gaussian_heatmaps, build_keypoints_gaussian_heatmaps,
    build_keypoints_heatmaps,
)
import logging
from segment_anything import SamPredictor, sam_model_registry

log = logging.getLogger(__name__)


class ReidDataset(ImageDataset):
    img_ext = ".jpg"
    masks_ext = ".npy"
    reid_dir = "bpbreid"
    reid_images_dir = "images"
    reid_masks_dir = "masks"
    reid_fig_dir = "figures"
    reid_anns_dir = "anns"
    images_anns_filename = "reid_crops_anns.json"
    masks_anns_filename = "reid_masks_anns.json"
    train_dir = 'gaussian_joints'
    dataset_dir = "PoseTrack21/reid"

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        "keypoints": (17, False, ".npy", ["p{}".format(p) for p in range(1, 17)],),
        "keypoints_gaussian": (17, False, ".npy", ["p{}".format(p) for p in range(1, 17)],),
        "joints": (10, False, ".npy", ["p{}".format(p) for p in range(1, 17)]),
        "joints_gaussian": (10, False, ".npy", ["p{}".format(p) for p in range(1, 17)]),
        "pose_on_img": (35, False, ".npy", ["p{}".format(p) for p in range(1, 35)]),
        "pose_on_img_crops": (35, False, ".npy", ["p{}".format(p) for p in range(1, 35)]),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in ReidDataset.masks_dirs:
            return None
        else:
            return ReidDataset.masks_dirs[masks_dir]

    def gallery_filter(self, q_pid, q_camid, q_ann, g_pids, g_camids, g_anns):
        """camid refers to video id: remove gallery samples from the different videos than query sample"""
        if self.eval_metric == "mot_inter_intra_video":
            return np.array(np.zeros_like(g_pids), dtype=bool)
        elif self.eval_metric == "mot_inter_video":
            remove = g_camids == q_camid
            return remove
        elif self.eval_metric == "mot_intra_video":
            remove = g_camids != q_camid
            return remove
        else:
            raise ValueError

    def __init__(
        self,
        tracking_dataset: TrackingDataset,
        reid_config,
        pose_model=None,
        masks_dir="",
        occluded_dataset=False,
        **kwargs
    ):
        # Init
        self.tracking_dataset = tracking_dataset
        self.reid_config = reid_config
        self.pose_model = pose_model
        self.dataset_path = Path(self.tracking_dataset.dataset_path)
        self.masks_dir = masks_dir
        self.pose_datapipe = EngineDatapipe(self.pose_model)
        self.pose_dl = None
        self.eval_metric = self.reid_config.eval_metric
        self.multi_video_queries_only = self.reid_config.multi_video_queries_only
        self.occluded_dataset = occluded_dataset
        self.sam_checkpoint = self.reid_config.sam_checkpoint

        assert (
            self.reid_config.train.max_samples_per_id
            >= self.reid_config.train.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"
        assert (
            self.reid_config.test.max_samples_per_id
            >= self.reid_config.test.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"

        if self.masks_dir in self.masks_dirs:  # TODO
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
            tracking_dataset.train_set,
            self.reid_config,
            is_test_set=False,
        )

        self.build_reid_set(
            tracking_dataset.val_set,
            self.reid_config,
            is_test_set=True,
        )

        train_gt_dets = tracking_dataset.train_set.detections_gt
        val_gt_dets = tracking_dataset.val_set.detections_gt

        # Get train/query/gallery sets as torchreid list format
        train_df = train_gt_dets[train_gt_dets["split"] == "train"]
        query_df = val_gt_dets[val_gt_dets["split"] == "query"]
        gallery_df = val_gt_dets[val_gt_dets["split"] == "gallery"]
        train, query, gallery = self.to_torchreid_dataset_format(
            [train_df, query_df, gallery_df]
        )

        super().__init__(train, query, gallery, **kwargs)

    def build_reid_set(self, tracking_set, reid_config, is_test_set):
        """
        Build ReID metadata for a given MOT dataset split.
        Only a subset of all MOT groundtruth detections is used for ReID.
        Detections to be used for ReID are selected according to the filtering criteria specified in the config 'reid_cfg'.
        Image crops and human parsing labels (masks) are generated for each selected detection only.
        If the config is changed and more detections are selected, the image crops and masks are generated only for
        these new detections.
        """
        split = tracking_set.split
        image_metadatas = tracking_set.image_metadatas
        detections = tracking_set.detections_gt
        detections["visibility"] = detections.keypoints.visibility()
        fig_size = reid_config.fig_size
        mask_size = reid_config.mask_size
        max_crop_size = reid_config.max_crop_size
        reid_set_cfg = reid_config.test if is_test_set else reid_config.train
        masks_mode = reid_config.masks_mode

        log.info("Loading {} set...".format(split))

        # Precompute all paths
        reid_path = Path(self.dataset_path, self.reid_dir, masks_mode)
        reid_img_path = reid_path / self.reid_images_dir / split
        reid_mask_path = reid_path / self.reid_masks_dir / split
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

        # Load reid crops metadata into existing ground truth detections dataframe
        self.load_reid_annotations(
            detections,
            reid_anns_filepath,
            ["reid_crop_path", "reid_crop_width", "reid_crop_height", "negative_kps"],
        )
        detections["negative_kps"] = detections["negative_kps"].apply(lambda x: np.array(x) if (isinstance(x, list) and len(x) > 0) else np.empty((0, 17, 3)))

        # Load reid masks metadata into existing ground truth detections dataframe
        self.load_reid_annotations(detections, masks_anns_filepath, ["masks_path"])
        # Sampling of detections to be used to create the ReID dataset
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
        self.save_reid_masks_crops(
            detections,
            reid_mask_path,
            reid_fig_path,
            split,
            masks_anns_filepath,
            image_metadatas,
            fig_size,
            mask_size,
            mode=masks_mode,
        )

        # Add 0-based pid column (for Torchreid compatibility) to sampled detections
        self.add_pid_column(detections)
        self.add_occlusion_level_column(detections)

        # Flag sampled detection as a query or gallery if this is a test set
        if is_test_set:
            self.query_gallery_split(detections, reid_set_cfg.ratio_query_per_id)

    def add_negative_samples(self, _df):
        _df["id"] = _df.index
        _df.reset_index(drop=True, inplace=True)
        all_kps_in_img = np.array(list(_df.keypoints_xyc))
        id_to_index = {k: v for v, k in enumerate(list(_df.id))}
        pass
        _df["negative_kps"] = _df\
            .apply(lambda bb: keypoints_in_bbox_coord(np.delete(all_kps_in_img, id_to_index[bb.id], axis=0), bb.bbox_ltwh), axis=1)\
            .apply(lambda kp_xyc_bbox: kp_xyc_bbox[kp_xyc_bbox[:, :, 2].sum(axis=1) > 0]) # remove non visibile skeletons
        _df.set_index("id", inplace=True)
        return _df

    def load_reid_annotations(self, gt_dets, reid_anns_filepath, columns):
        if reid_anns_filepath.exists():
            reid_anns = pd.read_json(
                reid_anns_filepath, convert_dates=False, convert_axes=False
            )
            reid_anns.index = reid_anns.index.astype(int)
            tmp_df = gt_dets.merge(
                reid_anns,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )
            gt_dets[columns] = tmp_df[columns]
        else:
            # no annotations yet, initialize empty columns
            for col in columns:
                gt_dets[col] = None

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
            .reset_index(level=0, drop=True)
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

        dets_df.loc[dets_df_f5.index, "split"] = "train"
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
                img_metadata = metadatas_df[metadatas_df.index == image_id].iloc[0]
                img = cv2.imread(img_metadata.file_path)
                for det_id, det_metadata in dets_from_img.iterrows():
                    # crop and resize bbox from image
                    l, t, w, h = det_metadata.bbox.ltwh(
                        image_shape=(img.shape[1], img.shape[0]), rounded=True
                    )
                    pid = det_metadata.person_id
                    img_crop = img[t : t + h, l : l + w]
                    if h > max_h or w > max_w:
                        img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)

                    # save crop to disk
                    filename = "{}_{}_{}{}".format(
                        pid, video_id, img_metadata.name, self.img_ext
                    )
                    rel_filepath = Path(str(video_id), filename)
                    abs_filepath = Path(save_path, rel_filepath)
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(abs_filepath), img_crop)

                    # save image crop metadata
                    gt_dets.at[det_metadata.name, "reid_crop_path"] = str(abs_filepath)
                    gt_dets.at[det_metadata.name, "reid_crop_width"] = img_crop.shape[1]
                    gt_dets.at[det_metadata.name, "reid_crop_height"] = img_crop.shape[0]
                    pbar.update(1)

        log.info(
            'Saving reid crops annotations as json to "{}"'.format(reid_anns_filepath)
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets["index"] = gt_dets.index
        gt_dets[
            ["index", "reid_crop_path", "reid_crop_width", "reid_crop_height", "negative_kps"]
        ].to_json(reid_anns_filepath)

    def save_reid_masks_crops(
        self,
        gt_dets,
        masks_save_path,
        fig_save_path,
        set_name,
        reid_anns_filepath,
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
        if mode == "pose_on_img_crops" or mode == "pose_on_img":
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
            sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
            predictor = SamPredictor(sam)
        if len(gt_dets_for_reid) == 0:
            log.info("All reid crops already have human parsing masks labels.")
            return
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} human parsing labels".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                img_metadata = metadatas_df[metadatas_df.index == image_id].iloc[0]
                # load image once to get video frame size
                if mode == "pose_on_img":
                    if self.pose_dl == None:
                        self.pose_dl = DataLoader(
                            dataset=self.pose_datapipe,
                            batch_size=128,
                            num_workers=0,
                            collate_fn=type(self.pose_model).collate_fn,
                            persistent_workers=False,
                        )
                    fields_list = []
                    self.pose_datapipe.update(
                        metadatas_df[metadatas_df.index == image_id], None
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
                for id, det_metadata in dets_from_img.iterrows():
                    img_crop = cv2.imread(det_metadata.reid_crop_path)
                    img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                    l, t, w, h = bbox_ltwh = det_metadata.bbox.ltwh(rounded=True)
                    kps_xyc_or = det_metadata.keypoints.in_bbox_coord(bbox_ltwh)
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
                    negative_kps_xyc = rescale_keypoints(negative_kps_xyc, (w, h), (fig_w, fig_h))

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
                        processed_image, anns, meta = self.pose_model.pifpaf_preprocess(pim_img_crop, [], {})  # FIXME size
                        processed_image = processed_image.unsqueeze(0)
                        _, fields_batch = self.pose_model.processor.batch(
                            self.pose_model.model, processed_image, device=self.pose_model.device
                        )
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

                        sam_mask = self.compute_sam_mask(predictor, img_crop, keypoints_xyc_crop, negative_kps_xyc)
                        masks_gt_crop = masks_gt_crop.squeeze().numpy()
                        kernel = np.ones((10, 10), np.uint8)
                        sam_mask = cv2.dilate(sam_mask.astype(np.uint8), kernel, iterations=2)

                        sam_mask = cv2.resize(sam_mask.squeeze(), (mask_w, mask_h))

                        masks_gt_crop = masks_gt_crop * sam_mask
                    elif mode == "pose_on_img":
                        # compute human parsing heatmaps using output of pose model on full image
                        l, t, w, h = det_metadata.bbox.ltwh(
                            image_shape=(img.shape[1], img.shape[0]), rounded=True
                        )
                        img_crop = img[t : t + h, l : l + w]
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        masks_gt_crop = masks_gt[:, t : t + h, l : l + w]
                        masks_gt_crop = resize(
                            masks_gt_crop, (masks_gt_crop.shape[0], fig_h, fig_w)
                        )
                        sam_mask = self.compute_sam_mask(predictor, img_crop, keypoints_xyc_crop, negative_kps_xyc)
                        masks_gt_crop = masks_gt_crop * sam_mask
                    else:
                        raise ValueError("Invalid human parsing method '{}'".format(mode))

                    # save human parsing heatmaps on disk
                    pid = det_metadata.person_id
                    filename = "{}_{}_{}".format(pid, video_id, image_id)
                    abs_filepath = Path(
                        masks_save_path, Path(str(video_id), filename + self.masks_ext)
                    )
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(abs_filepath), masks_gt_crop)

                    # save image crop with human parsing heatmaps overlayed on disk for visualization/debug purpose
                    img_with_heatmap = colored_body_parts_overlay(
                        img_crop, masks_gt_crop
                    )
                    figure_filepath = Path(
                        fig_save_path, str(video_id), filename + "_heatmaps_" + self.img_ext
                    )
                    figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(figure_filepath), img_with_heatmap)
                    img_crop_kps = draw_keypoints(img_crop, keypoints_xyc_crop, (fig_w, fig_h), radius=2, thickness=2, color=(255, 102, 255))
                    for negative_kps in negative_kps_xyc:
                        img_crop_kps = draw_keypoints(img_crop_kps, negative_kps, (fig_w, fig_h), radius=2, thickness=2, color=(0, 0, 255))
                    kps_filepath = Path(
                        fig_save_path, str(video_id), filename + "_kps_"  + self.img_ext
                    )
                    cv2.imwrite(str(kps_filepath), img_crop_kps)
                    # record human parsing metadata for later json dump
                    gt_dets.at[det_metadata.name, "masks_path"] = str(abs_filepath)
                    pbar.update(1)

        log.info(
            'Saving reid human parsing annotations as json to "{}"'.format(
                reid_anns_filepath
            )
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets["id"] = gt_dets.index
        gt_dets[["id", "masks_path"]].to_json(reid_anns_filepath)

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
        ).reset_index(level=0, drop=True)
        if self.eval_metric == "mot_inter_video" or self.multi_video_queries_only:
            # keep only queries that are in more than one video
            queries_per_pid = (
                queries_per_pid.droplevel(level=0)
                .groupby("person_id")["video_id"]
                .filter(lambda g: (g.nunique() > 1))
                .reset_index(level=0)
            )
            assert len(queries_per_pid) != 0, (
                "There were no identity with more than one videos to be used as queries. "
                "Try setting 'multi_video_queries_only' to False or not using "
                "eval_metric='mot_inter_video' or adjust the settings to sample a "
                "bigger ReID dataset."
            )
        gt_dets.loc[gt_dets.split != "none", "split"] = "gallery"
        gt_dets.loc[queries_per_pid.index, "split"] = "query"

    def to_torchreid_dataset_format(self, dataframes):
        results = []
        for df in dataframes:
            df = df.copy()  # to avoid SettingWithCopyWarning
            # use video id as camera id: camid is used at inference to filter out gallery samples given a query sample
            df["camid"] = pd.Categorical(df.video_id, categories=df.video_id.unique()).codes
            df["img_path"] = df["reid_crop_path"]
            df["keypoints_xyc"] = df.keypoints.keypoints_bbox_xyc()
            df["keypoints_xyc"] = df.apply(lambda r: rescale_keypoints(r.keypoints_xyc, (r.bbox_ltwh[2], r.bbox_ltwh[3]), (r.reid_crop_width, r.reid_crop_height)), axis=1)
            df["negative_kps"] = df.apply(lambda r: rescale_keypoints(r.negative_kps, (r.bbox_ltwh[2], r.bbox_ltwh[3]), (r.reid_crop_width, r.reid_crop_height)), axis=1)

            # remove bbox_head as it is not available for each sample
            # df to list of dict
            sorted_df = df.sort_values(by=["pid"])
            # use only necessary annotations: using them all caused a
            # 'RuntimeError: torch.cat(): input types can't be cast to the desired output type Long' in collate.py
            # -> still has to be fixed
            data_list = sorted_df[
                ["pid", "camid", "video_id", "img_path", "masks_path", "visibility", "keypoints_xyc", "reid_crop_width", "reid_crop_height", "negative_kps", "occ_level"]
            ]
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


def draw_keypoints(img_to_insert, kp, model_img_size, radius=2, thickness=2, vis_thresh=0, color=None):
    kp = rescale_keypoints(kp, model_img_size, (img_to_insert.shape[1], img_to_insert.shape[0]))
    for xyc in kp:
        x, y, c = xyc
        if c > 0:
            if color is not None:
                match_color = color
                kp_thickness = thickness
                kp_radius = radius
            else:
                if c > vis_thresh:
                    match_color = cmap(c/3, bytes=True)[0:-1]  # divided by three because hsv colormap goes from red to green inside [0, 0.333]
                    match_color = (int(match_color[2]), int(match_color[1]), int(match_color[0]))
                    kp_thickness = thickness
                    kp_radius = radius
                else:
                    match_color = BLACK
                    kp_thickness = thickness-1
                    kp_radius = radius-1
            cv2.circle(
                img_to_insert,
                (int(x), int(y)),
                color=match_color,
                thickness=kp_thickness,
                radius=kp_radius,
                lineType=cv2.LINE_AA,
            )
    return img_to_insert
