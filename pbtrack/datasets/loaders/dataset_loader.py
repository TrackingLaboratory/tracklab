
from __future__ import absolute_import, division, print_function

import os.path as osp
import sys
from abc import ABC, abstractmethod
from math import ceil
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from os import path as osp
from pathlib import Path

import torch
from skimage.transform import resize
from tqdm import tqdm

from pbtrack.tracker.categories import Categories
from pbtrack.tracker.detections import Detections
from pbtrack.tracker.images import Images
from pbtrack.tracker.tracker import Tracker
from pbtrack.utils.coordinates import (
    rescale_keypoints,
    clip_to_img_dim,
)
from pbtrack.utils.images import overlay_heatmap
from hydra.utils import to_absolute_path

sys.path.append(to_absolute_path("modules/reid/bpbreid"))
from torchreid.utils.imagetools import gkern, build_gaussian_heatmaps


class DatasetLoader(ABC):
    splits = {}

    # eval_metric = "mot_inter_video"
    # img_ext = ".jpg"
    # masks_ext = ".npy"
    # reid_dir = "reid"
    # reid_images_dir = "images"
    # reid_masks_dir = "masks"
    # reid_fig_dir = "figures"
    # reid_anns_dir = "anns"
    # images_anns_filename = "reid_crops_anns.json"
    # masks_anns_filename = "reid_masks_anns.json"

    def __init__(
        self,
        # reid_cfg,
        dataset_path='',
        # pose_model=None,
    ):
        # self.reid_cfg = reid_cfg
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), "dataset_path '{}' does not exist".format(
            self.dataset_path
        )
        # self.pose_model = pose_model
        # assert (
        #     self.reid_cfg.train.max_samples_per_id >= self.reid_cfg.train.min_samples_per_id
        # ), "max_samples_per_id must be >= min_samples_per_id"
        # assert (
        #     self.reid_cfg.test.max_samples_per_id >= self.reid_cfg.test.min_samples_per_id
        # ), "max_samples_per_id must be >= min_samples_per_id"

    @abstractmethod
    def load_dataset(self) -> Tuple[Images, Categories, Detections]:
        """
        Load all train/test/val dataset annotations and return images, categories and detections dataframes
        in correct format, as specified in pbtrack.tracker.images.Images, pbtrack.tracker.categories.Categories and
        pbtrack.tracker.detections.Detections.
        """
        pass

    def build_tracker(self):
        images, categories, detections = self.load_dataset()
        assert isinstance(images, Images), "images must be an instance of Images"
        assert isinstance(categories, Categories), "categories must be an instance of Categories"
        assert isinstance(detections, Detections), "detections must be an instance of Detections"
        return Tracker(images, categories, detections)

    # def build_tracker(self):
    #     images, categories, detections = self.load_dataset()
    #     assert isinstance(images, Images), "images must be an instance of Images"
    #     assert isinstance(categories, Categories), "categories must be an instance of Categories"
    #     assert isinstance(detections, Detections), "detections must be an instance of Detections"
    #     # add reid metadata for each split
    #     for split, info in self.splits.items():
    #         build_reid_meta = info["build_reid_meta"]
    #         is_test_set = info["is_test_set"]
    #         if build_reid_meta:
    #             images_split = images[images['split'] == split]
    #             detections_split = detections[detections['split'] == split]
    #             self.add_reid_metadata(
    #                 images_split,
    #                 detections_split,
    #                 self.reid_cfg,
    #                 split,
    #                 is_test_set=is_test_set,
    #             )
    #     return Tracker(images, categories, detections)

    def add_reid_metadata(
        self, images, detections, reid_cfg, split, is_test_set
    ):
        """
        Build ReID metadata for a given MOT dataset split.
        Only a subset of all MOT groundtruth detections is used for ReID.
        Detections to be used for ReID are selected according to the filtering criteria specified in the config 'reid_cfg'.
        Image crops and human parsing labels (masks) are generated for each selected detection only.
        If the config is changed and more detections are selected, the image crops and masks are generated only for
        these new detections.
        """
        fig_size = reid_cfg.fig_size
        mask_size = reid_cfg.mask_size
        max_crop_size = reid_cfg.max_crop_size
        reid_cfg = reid_cfg.test if is_test_set else reid_cfg.train

        print("Loading {} set...".format(split))

        # Precompute all paths
        reid_path = Path(self.dataset_path, self.reid_dir)
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
        detections = self.load_reid_annotations(
            detections,
            reid_anns_filepath,
            ["reid_crop_path", "reid_crop_width", "reid_crop_height"],
        )

        # Load reid masks metadata into existing ground truth detections dataframe
        detections = self.load_reid_annotations(
            detections, masks_anns_filepath, ["masks_path"]
        )

        # Sampling of detections to be used to create the ReID dataset
        self.sample_detections_for_reid(detections, reid_cfg)

        # Save ReID detections crops and related metadata. Apply only on sampled detections
        self.save_reid_img_crops(
            detections, reid_img_path, split, reid_anns_filepath, images, max_crop_size
        )

        # Save human parsing pseudo ground truth and related metadata. Apply only on sampled detections
        self.save_reid_masks_crops(
            detections,
            reid_mask_path,
            reid_fig_path,
            split,
            masks_anns_filepath,
            images,
            fig_size,
            mask_size,
        )

        # Add 0-based pid column (for Torchreid compatibility) to sampled detections
        self.ad_pid_column(detections)

        # Flag sampled detection as a query or gallery if this is a test set
        if is_test_set:
            self.query_gallery_split(detections, reid_cfg.ratio_query_per_id)

    def load_reid_annotations(self, gt_dets, reid_anns_filepath, columns):
        if reid_anns_filepath.exists():
            reid_anns = pd.read_json(reid_anns_filepath)
            assert len(reid_anns) == len(gt_dets), (
                "reid_anns_filepath and gt_dets must have the same length. Delete "
                "'{}' and re-run the script.".format(reid_anns_filepath)
            )
            return gt_dets.merge(
                reid_anns,
                left_index=False,
                right_index=False,
                left_on="id",
                right_on="id",
                validate="one_to_one",
            )
            # merged_df.drop('index', axis=1, inplace=True)
        else:
            # no annotations yet, initialize empty columns
            for col in columns:
                gt_dets[col] = None
            return gt_dets

    def sample_detections_for_reid(self, dets_df, reid_cfg):
        dets_df["split"] = "none"

        # Filter detections by visibility
        dets_df_f1 = dets_df[dets_df.visibility >= reid_cfg.min_vis]

        # Filter detections by crop size
        keep = dets_df_f1.bbox_ltwh.apply(
            lambda x: x[2] > reid_cfg.min_w
        ) & dets_df_f1.bbox_ltwh.apply(lambda x: x[3] > reid_cfg.min_h)
        dets_df_f2 = dets_df_f1[keep]
        print(
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
            .apply(self.uniform_tracklet_sampling, reid_cfg.max_samples_per_id, "image_id")
            .reset_index(drop=True)
            .copy()
        )
        print(
            "{} removed for uniform tracklet sampling = {}".format(
                self.__class__.__name__, len(dets_df_f2) - len(dets_df_f3)
            )
        )

        # Keep only ids with at least MIN_SAMPLES appearances
        count_per_id = dets_df_f3.person_id.value_counts()
        ids_to_keep = count_per_id.index[count_per_id.ge((reid_cfg.min_samples_per_id))]
        dets_df_f4 = dets_df_f3[dets_df_f3.person_id.isin(ids_to_keep)]
        print(
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
        print("{} filtered size = {}".format(self.__class__.__name__, len(dets_df_f5)))

    def save_reid_img_crops(
        self, gt_dets, save_path, set_name, reid_anns_filepath, images_df, max_crop_size
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
            print(
                "All detections used for ReID already have their image crop saved on disk."
            )
            return
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} reid crops".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                img_metadata = images_df[images_df.image_id == image_id].iloc[0]
                filename = img_metadata.file_name
                img = cv2.imread(str(self.dataset_path / filename))
                for det_metadata in dets_from_img.itertuples():
                    # crop and resize bbox from image
                    bbox_ltwh = det_metadata.bbox_ltwh
                    bbox_ltwh = clip_to_img_dim(bbox_ltwh, img.shape[1], img.shape[0])
                    pid = det_metadata.person_id
                    l, t, w, h = bbox_ltwh.astype(int)
                    img_crop = img[t : t + h, l : l + w]
                    if h > max_h or w > max_w:
                        img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)

                    # save crop to disk
                    filename = "{}_{}_{}{}".format(
                        pid, video_id, img_metadata.image_id, self.img_ext
                    )
                    rel_filepath = Path(video_id, filename)
                    abs_filepath = Path(save_path, rel_filepath)
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(abs_filepath), img_crop)

                    # save image crop metadata
                    gt_dets.at[det_metadata.Index, "reid_crop_path"] = str(abs_filepath)
                    gt_dets.at[det_metadata.Index, "reid_crop_width"] = img_crop.shape[
                        0
                    ]
                    gt_dets.at[det_metadata.Index, "reid_crop_height"] = img_crop.shape[
                        1
                    ]
                    pbar.update(1)

        print(
            'Saving reid crops annotations as json to "{}"'.format(reid_anns_filepath)
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets[
            ["id", "reid_crop_path", "reid_crop_width", "reid_crop_height"]
        ].to_json(reid_anns_filepath)

    def save_reid_masks_crops(
        self,
        gt_dets,
        masks_save_path,
        fig_save_path,
        set_name,
        reid_anns_filepath,
        images_df,
        fig_size,
        masks_size,
        mode="gaussian_keypoints",
    ):
        """
        Save on disk all human parsing gt for each reid crop.
        Create a json annotation file with human parsing metadata.
        """
        fig_h, fig_w = fig_size
        mask_h, mask_w = masks_size
        g_scale = 6
        g_radius = int(mask_w / g_scale)
        gaussian = gkern(g_radius * 2 + 1)
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.masks_path.isnull()
        ]
        if len(gt_dets_for_reid) == 0:
            print("All reid crops already have human parsing masks labels.")
            return
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} human parsing labels".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                img_metadata = images_df[images_df.image_id == image_id].iloc[0]
                filename = img_metadata.file_name
                # load image once to get video frame size
                if mode == "pose_on_img":
                    img = cv2.imread(str(self.dataset_path / filename))
                    _, masks_gt_or = self.pose_model.run(
                        torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0)
                    )  # TODO check if pose_model need BRG or RGB
                    masks_gt_or = (
                        masks_gt_or.squeeze(0).permute((1, 2, 0)).numpy()
                    )  # TODO why that permute needed? for old resize?
                    masks_gt = resize(
                        masks_gt_or, (img.shape[0], img.shape[1], masks_gt_or.shape[2])
                    )
                # loop on detections in frame
                for det_metadata in dets_from_img.itertuples():
                    if mode == "gaussian_keypoints":
                        # compute human parsing heatmaps as gaussian on each visible keypoint
                        img_crop = cv2.imread(det_metadata.reid_crop_path)
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        l, t, w, h = det_metadata.bbox_ltwh
                        keypoints_xyc = rescale_keypoints(
                            det_metadata.keypoints_bbox_xyc, (w, h), (mask_w, mask_h)
                        )
                        masks_gt_crop = build_gaussian_heatmaps(
                            keypoints_xyc, mask_w, mask_h, gaussian=gaussian
                        )
                    elif mode == "pose_on_img_crops":
                        # compute human parsing heatmaps using output of pose model on cropped person image
                        img_crop = cv2.imread(det_metadata.reid_crop_path)
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        _, masks_gt_crop = self.pose_model.run(
                            torch.from_numpy(img_crop).permute((2, 0, 1)).unsqueeze(0)
                        )
                        masks_gt_crop = (
                            masks_gt_crop.squeeze().permute((1, 2, 0)).numpy()
                        )
                        masks_gt_crop = resize(
                            masks_gt_crop, (fig_h, fig_w, masks_gt_crop.shape[2])
                        )
                    elif mode == "pose_on_img":
                        # compute human parsing heatmaps using output of pose model on full image
                        bbox_ltwh = clip_to_img_dim(
                            det_metadata.bbox_ltwh, img.shape[1], img.shape[0]
                        ).astype(int)
                        l, t, w, h = bbox_ltwh
                        img_crop = img[t : t + h, l : l + w]
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        masks_gt_crop = masks_gt[t : t + h, l : l + w]
                        masks_gt_crop = resize(
                            masks_gt_crop, (fig_h, fig_w, masks_gt_crop.shape[2])
                        )
                    else:
                        raise ValueError("Invalid human parsing method")

                    # save human parsing heatmaps on disk
                    pid = det_metadata.person_id
                    filename = "{}_{}_{}".format(pid, video_id, image_id)
                    abs_filepath = Path(
                        masks_save_path, Path(video_id, filename + self.masks_ext)
                    )
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(abs_filepath), masks_gt_crop)

                    # save image crop with human parsing heatmaps overlayed on disk for visualization/debug purpose
                    img_with_heatmap = overlay_heatmap(
                        img_crop, masks_gt_crop.max(axis=0), weight=0.3
                    )
                    figure_filepath = Path(
                        fig_save_path, video_id, filename + self.img_ext
                    )
                    figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(figure_filepath), img_with_heatmap)

                    # record human parsing metadata for later json dump
                    gt_dets.at[det_metadata.Index, "masks_path"] = str(abs_filepath)
                    pbar.update(1)

        print(
            'Saving reid human parsing annotations as json to "{}"'.format(
                reid_anns_filepath
            )
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets[["id", "masks_path"]].to_json(reid_anns_filepath)

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

        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        # reset seed to make sure the same split is used if the dataset is instantiated multiple times
        np.random.seed(0)
        queries_per_pid = gt_dets_for_reid.groupby("person_id").apply(
            random_tracklet_sampling
        )
        gt_dets.loc[gt_dets.split != "none", "split"] = "gallery"
        gt_dets.loc[gt_dets.id.isin(queries_per_pid.id), "split"] = "query"

    def to_torchreid_dataset_format(self, dataframes):
        results = []
        for df in dataframes:
            df = df.copy()  # to avoid SettingWithCopyWarning
            # use video id as camera id: camid is used at inference to filter out gallery samples given a query sample
            df["camid"] = df["video_id"]
            df["img_path"] = df["reid_crop_path"]
            # remove bbox_head as it is not available for each sample
            df.drop(columns="bbox_head", inplace=True)
            # df to list of dict
            data_list = df.sort_values(by=["pid"])
            # use only necessary annotations: using them all caused a
            # 'RuntimeError: torch.cat(): input types can't be cast to the desired output type Long' in collate.py
            # -> still has to be fixed
            data_list = data_list[["pid", "camid", "img_path", "masks_path"]]
            data_list = data_list.to_dict("records")
            results.append(data_list)
        return results

    def ad_pid_column(self, gt_dets):
        # create pids as 0-based increasing numbers
        gt_dets["pid"] = None
        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        gt_dets.loc[gt_dets_for_reid.index, "pid"] = pd.factorize(
            gt_dets_for_reid.person_id
        )[0]

    def uniform_tracklet_sampling(self, _df, max_samples_per_id, column):
        _df.sort_values(column)
        num_det = len(_df)
        if num_det > max_samples_per_id:
            # Select 'max_samples_per_id' evenly spaced indices, including first and last
            indices = np.round(np.linspace(0, num_det - 1, max_samples_per_id)).astype(int)
            assert len(indices) == max_samples_per_id
            return _df.iloc[indices]
        else:
            return _df
