from __future__ import absolute_import, division, print_function

import json
import os.path as osp
from math import ceil

import cv2
import numpy as np
import pandas as pd

from os import path as osp
from pathlib import Path

import torch
from scipy import signal
from skimage.transform import resize
from tqdm import tqdm

from reconnaissance.utils.coordinates import kp_img_to_kp_bbox, rescale_keypoints, clip_to_img_dim
from reconnaissance.utils.images import overlay_heatmap
from torchreid.data import ImageDataset


# TODO run it on Belldev
# ----
# TODO filter FIRST: generate reid data only for filtered detections
# TODO get changes from other BPBreID branch
# TODO bootstrap general purpose dataset class for MOT datasets (will be improved later when integrating with other datasets)
# TODO print stats from posetrack test
# todo add nice prints + docstring
# TODO load HRNet and other pretrained weights from URLs
# TODO cfg.data.masks_dir not used + refactor folder structure
# TODO fix 'none' in masks_preprocess_transforms: should be able to use none to indicate to use raw masks, load_masks should come from 'disk' vs 'stripes'
# ----
# TODO add pifpaf to the pipeline
# TODO batch processing of heatmaps
# TODO make sure format is good: RGB vs BGR, etc
# TODO make this class dataset independant
# TODO: make different config for query and gallery?


def uniform_tracklet_sampling(_df, max_samples_per_id, column):
    _df.sort_values(column)
    num_det = len(_df)
    if num_det > max_samples_per_id:
        # Select 'max_samples_per_id' evenly spaced indices, including first and last
        indices = np.round(np.linspace(0, num_det - 1, max_samples_per_id)).astype(int)
        assert len(indices) == max_samples_per_id
        return _df.iloc[indices]
    else:
        return _df


def ad_pid_column(df):
    # create pids as 0-based increasing numbers
    df['pid'] = pd.factorize(df.person_id)[0]
    return df


def random_sampling_per_pid(df, ratio=1.0):
    def uniform_tracklet_sampling(_df):
        x = list(_df.index)
        size = ceil(len(x) * ratio)
        result = list(np.random.choice(x, size=size, replace=False))
        return _df.loc[result]

    per_pid = df.groupby('person_id').apply(uniform_tracklet_sampling)
    queries = df[df.id.isin(per_pid.id)]
    galleries = df[~df.id.isin(per_pid.id)]
    return queries, galleries


class PoseTrack21ReID(ImageDataset):
    dataset_dir = 'Posetrack21'
    eval_metric = 'mot_inter_video'
    annotations_dir = 'posetrack_data'
    img_ext = '.jpg'
    masks_ext = '.npy'

    reid_dir = 'reid'
    reid_images_dir = 'images'
    reid_masks_dir = 'masks'
    reid_fig_dir = 'figures'
    reid_anns_dir = 'anns'
    images_anns_filename = 'reid_crops_anns.json'
    masks_anns_filename = 'reid_masks_anns.json'

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        'gaussian': (17, False, '.npy', ["p{}".format(p) for p in range(1, 5+1)]),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in PoseTrack21ReID.masks_dirs:
            return None
        else:
            return PoseTrack21ReID.masks_dirs[masks_dir]

    def __init__(self, config, datasets_root='', masks_dir='', crop_dim=(384, 128), pose_model=None, **kwargs):

        # Init
        self.config = config
        mot_cfg = config.data.mot
        datasets_root = Path(osp.abspath(osp.expanduser(datasets_root)))
        self.dataset_dir = Path(datasets_root, self.dataset_dir)
        self.pose_model = pose_model
        assert mot_cfg.train.max_samples_per_id >= mot_cfg.train.min_samples_per_id, "max_samples_per_id must be >= min_samples_per_id"
        assert mot_cfg.test.max_samples_per_id >= mot_cfg.test.min_samples_per_id, "max_samples_per_id must be >= min_samples_per_id"
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix, self.masks_parts_names = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix, self.masks_parts_names = None, None, None, None

        # Load Posetrack21 dataset and build ReID variant
        self.train_gt_dets, self.train_images, self.train_categories = self.build_dataset(crop_dim, mot_cfg.train, 'train', is_test_set=False)
        self.val_gt_dets, self.val_images, self.val_categories = self.build_dataset(crop_dim, mot_cfg.test, 'val', is_test_set=False)

        # Get train/query/gallery sets as torchreid list format
        train_df = self.train_gt_dets
        query_df = self.val_gt_dets[self.val_gt_dets['split'] == 'query']
        gallery_df = self.val_gt_dets[self.val_gt_dets['split'] == 'gallery']
        train, query, gallery = self.to_torchreid_dataset_format([train_df, query_df, gallery_df])

        super(PoseTrack21ReID, self).__init__(train, query, gallery, **kwargs)

    def build_dataset(self, crop_dim, mot_cfg, set_name, is_test_set):
        # anns_path reid_set_path
        crop_size = (128, 64)  # vs crop_dim
        mask_size = (32, 16)

        # Precompute all paths
        anns_path = Path(self.dataset_dir, self.annotations_dir, set_name)
        reid_path = Path(self.dataset_dir, self.reid_dir)
        reid_img_path = reid_path / self.reid_images_dir / set_name
        reid_mask_path = reid_path / self.reid_masks_dir / set_name
        reid_fig_path = reid_path / self.reid_fig_dir / set_name
        reid_anns_filepath = reid_img_path / self.reid_anns_dir / (set_name + '_' + self.images_anns_filename)

        # Load annotations into Pandas dataframes
        categories, gt_dets, images = self.build_annotations_df(anns_path)

        # Save detections crops and related metadata on disk to build ReID dataset
        existing_files = list(reid_img_path.glob('*/*{}'.format(self.img_ext)))
        if len(existing_files) != len(gt_dets) or not reid_anns_filepath.exists():
            self.build_reid_set(reid_img_path, set_name, reid_anns_filepath, gt_dets, images, crop_dim)

        # Load reid crops metadata into existing ground truth detections dataframe
        gt_dets = self.load_reid_annotations(reid_anns_filepath, gt_dets)

        # Build human parsing pseudo ground truth using the pose model
        existing_files = list(reid_mask_path.glob('*/*{}'.format(self.masks_ext)))
        masks_anns_filepath = reid_mask_path / self.reid_anns_dir / (set_name + '_' + self.masks_anns_filename)
        if len(existing_files) != len(gt_dets) or not masks_anns_filepath.exists():
            self.build_human_parsing_gt(reid_mask_path, reid_fig_path, set_name, masks_anns_filepath, gt_dets, images,
                                        crop_size, mask_size)

        # Load reid masks metadata into existing ground truth detections dataframe
        gt_dets = self.load_reid_annotations(masks_anns_filepath, gt_dets)
        gt_dets = self.filter_reid_samples(gt_dets,
                                      max_total_ids=mot_cfg.max_total_ids,
                                      min_vis=mot_cfg.min_vis,
                                      min_h=mot_cfg.min_h,
                                      min_w=mot_cfg.min_w,
                                      min_samples=mot_cfg.min_samples_per_id,
                                      max_samples_per_id=mot_cfg.max_samples_per_id)

        # Add 0-based pid column for Torchreid compatibility
        gt_dets = ad_pid_column(gt_dets)

        # Flag each detection as a query or gallery sample
        if is_test_set:
            gt_dets = self.query_gallery_split(gt_dets, mot_cfg.ratio_query_per_id)

        return gt_dets, images, categories

    def to_torchreid_dataset_format(self, dataframes):
        results = []
        for df in dataframes:
            df = df.copy()  # to avoid SettingWithCopyWarning
            # use video id as camera id: camid is used at inference to filter out gallery samples given a query sample
            df['camid'] = df['video_id']
            df['img_path'] = df['reid_crop_path']
            # remove bbox_head as it is not available for each sample
            df.drop(columns='bbox_head', inplace=True)
            # df to list of dict
            data_list = df.to_dict('records')
            results.append(data_list)
        return results

    def build_annotations_df(self, anns_path):
        annotations_list = []
        categories_list = []
        images_list = []
        anns_files_list = list(anns_path.glob('*.json'))
        assert len(anns_files_list) > 0, 'No annotations files found in {}'.format(anns_path)
        for path in anns_files_list:
            json_file = open(path)
            data_dict = json.load(json_file)
            annotations_list.append(pd.DataFrame(data_dict['annotations']))
            categories_list.append(pd.DataFrame(data_dict['categories']))
            images_list.append(pd.DataFrame(data_dict['images']))
        gt_dets = pd.concat(annotations_list).reset_index(drop=True)
        categories = pd.concat(categories_list).reset_index(drop=True)
        images = pd.concat(images_list).reset_index(drop=True)

        gt_dets.bbox = gt_dets.bbox.apply(lambda x: np.array(x))
        gt_dets.bbox_head = gt_dets.bbox_head.apply(lambda x: np.array(x))

        # reshape keypoints to (n, 3) array
        gt_dets.keypoints = gt_dets.keypoints.apply(lambda kp: np.array(kp).reshape(-1, 3))

        # keep global index as reference for further slicing operations
        gt_dets['global_index'] = gt_dets.index

        # rename base 'vid_id' to 'video_id', to be consistent with 'image_id'
        images.rename(columns={'vid_id': 'video_id'}, inplace=True)

        # add video_id to gt_dets, will be used for torchreid 'camid' paremeter
        gt_dets = gt_dets.merge(images[['image_id', 'video_id']], on='image_id', how='left')

        # compute detection visiblity as average keypoints visibility
        gt_dets['visibility'] = gt_dets.keypoints.apply(lambda x: x[:, 2].mean())

        # precompute various bbox formats
        gt_dets.rename(columns={'bbox': 'bbox_ltwh'}, inplace=True)
        gt_dets['bbox_ltrb'] = gt_dets.bbox_ltwh.apply(lambda ltwh: np.concatenate((ltwh[:2], ltwh[:2]+ltwh[2:])))
        gt_dets['bbox_cxcywh'] = gt_dets.bbox_ltwh.apply(lambda ltwh: np.concatenate((ltwh[:2]+ltwh[2:]/2, ltwh[2:])))

        # precompute various keypoints formats
        gt_dets.rename(columns={'keypoints': 'keypoints_xyc'}, inplace=True)
        gt_dets['keypoints_bbox_xyc'] = gt_dets.apply(lambda r: kp_img_to_kp_bbox(r.keypoints_xyc, r.bbox_ltwh), axis=1)

        return categories, gt_dets, images


    def build_reid_set(self, save_path, set_name, reid_anns_filepath, gt_dets_df, images_df, crop_dim):
        """
        Save on disk all detections image crops from the ground truth dataset to build the reid dataset.
        Create a json annotation file with crops metadata.
        """
        save_path = save_path / set_name
        max_h, max_w = crop_dim
        reid_crops_anns = []
        with tqdm(total=len(gt_dets_df)) as pbar:
            pbar.set_description('Extracting all {} bboxes crops'.format(set_name))
            # loop on videos
            for video_id in images_df.video_id.unique():
                # loop on video frames
                for img_metadata in images_df[images_df.video_id == video_id].itertuples():
                    if not img_metadata.is_labeled:
                        continue
                    filename = img_metadata.file_name
                    img = cv2.imread(str(self.dataset_dir / filename))
                    # loop on detections in frame
                    for det_metadata in gt_dets_df[gt_dets_df.image_id == img_metadata.image_id].itertuples():
                        # crop and resize bbox from image
                        bbox_ltwh = np.array(det_metadata.bbox_ltwh)
                        bbox_ltwh = clip_to_img_dim(bbox_ltwh, img.shape[1], img.shape[0])
                        pid = det_metadata.person_id
                        l, t, w, h = bbox_ltwh.astype(int)
                        img_crop = img[t:t + h, l:l + w]
                        if h > max_h or w > max_w:
                            img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)

                        # save crop to disk
                        filename = "{}_{}_{}{}".format(pid, video_id, img_metadata.image_id, self.img_ext)
                        rel_filepath = Path(video_id, filename)
                        abs_filepath = Path(save_path, rel_filepath)
                        abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(abs_filepath), img_crop)

                        # record image crop metadata for later json dump
                        reid_crops_anns.append({
                            'index': det_metadata.global_index,
                            'reid_crop_path': str(abs_filepath),
                            'reid_crop_width': img_crop.shape[0],
                            'reid_crop_height': img_crop.shape[1],
                        })
                        pbar.update(1)

        print('Saving reid crops annotations as json to "{}"'.format(reid_anns_filepath))
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(reid_crops_anns).to_json(reid_anns_filepath)

    def build_human_parsing_gt(self, masks_save_path, fig_save_path, set_name, reid_anns_filepath, gt_dets_df, images_df, crop_dim, masks_dim, mode='gaussian_keypoints'):
        """
        Save on disk all human parsing gt for each reid crop.
        Create a json annotation file with human parsing metadata.
        """
        crop_h, crop_w = crop_dim
        mask_h, mask_w = masks_dim
        reid_masks_anns = []
        g_scale = 6
        g_radius = int(mask_w / g_scale)
        gaussian = self.gkern(g_radius * 2 + 1)
        with tqdm(total=len(gt_dets_df)) as pbar:
            pbar.set_description('Extracting all {} human parsing labels'.format(set_name))
            # loop on videos
            for video_id in images_df.video_id.unique():
                img_shape = None
                # loop on video frames
                for img_metadata in images_df[images_df.video_id == video_id].itertuples():
                    filename = img_metadata.file_name
                    img_id = img_metadata.image_id
                    img_detections = gt_dets_df[gt_dets_df.image_id == img_id]
                    if len(img_detections) == 0:
                        continue
                    if img_shape is None:
                        # load image once to get video frame size
                        img_shape = cv2.imread(str(self.dataset_dir / filename)).shape
                    if mode == 'pose_on_img':
                        img = cv2.imread(str(self.dataset_dir / filename))
                        _, masks_gt_or = self.pose_model.run(torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0))  # TODO check if pose_model need BRG or RGB
                        masks_gt_or = masks_gt_or.squeeze(0).permute((1, 2, 0)).numpy()  # TODO why that permute needed? for old resize?
                        masks_gt = resize(masks_gt_or, (img.shape[0], img.shape[1], masks_gt_or.shape[2]))
                    # loop on detections in frame
                    for det_metadata in img_detections.itertuples():
                        bbox_ltwh = clip_to_img_dim(det_metadata.bbox_ltwh, img_shape[1], img_shape[0]).astype(int)
                        l, t, w, h = bbox_ltwh
                        if mode == 'gaussian_keypoints':
                            # compute human parsing heatmaps as gaussian on each visible keypoint
                            img_crop = cv2.imread(det_metadata.reid_crop_path)
                            img_crop = cv2.resize(img_crop, (crop_w, crop_h), cv2.INTER_CUBIC)
                            bbox_xyc = rescale_keypoints(det_metadata.keypoints_bbox_xyc, (w, h), (mask_w, mask_h))
                            masks_gt_crop = self.build_gaussian_heatmaps(bbox_xyc, len(det_metadata.keypoints_xyc), mask_w,
                                                                      mask_h, gaussian=gaussian)
                        elif mode == 'pose_on_img_crops':
                            # compute human parsing heatmaps using output of pose model on full image
                            img_crop = cv2.imread(det_metadata.reid_crop_path)
                            img_crop = cv2.resize(img_crop, (crop_w, crop_h), cv2.INTER_CUBIC)
                            _, masks_gt_crop = self.pose_model.run(torch.from_numpy(img_crop).permute((2, 0, 1)).unsqueeze(0))
                            masks_gt_crop = masks_gt_crop.squeeze().permute((1, 2, 0)).numpy()
                            masks_gt_crop = resize(masks_gt_crop, (crop_h, crop_w, masks_gt_crop.shape[2]))
                        elif mode == 'pose_on_img':
                            # compute human parsing heatmaps using output of pose model on cropped person image
                            img_crop = img[t:t + h, l:l + w]
                            img_crop = cv2.resize(img_crop, (crop_w, crop_h), cv2.INTER_CUBIC)
                            masks_gt_crop = masks_gt[t:t + h, l:l + w]
                            masks_gt_crop = resize(masks_gt_crop, (crop_h, crop_w, masks_gt_crop.shape[2]))
                        else:
                            raise ValueError('Invalid human parsing method')

                        # save human parsing heatmaps on disk
                        pid = det_metadata.person_id
                        filename = "{}_{}_{}".format(pid, video_id, img_id)
                        abs_filepath = Path(masks_save_path, Path(video_id, filename + self.masks_ext))
                        abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(abs_filepath), masks_gt_crop)

                        # save image crop with human parsing heatmaps overlayed on disk for visualization/debug purpose
                        img_with_heatmap = overlay_heatmap(img_crop, masks_gt_crop.max(axis=0), weight=0.3)
                        figure_filepath = Path(fig_save_path, video_id, filename + self.img_ext)
                        figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(figure_filepath), img_with_heatmap)

                        # record human parsing metadata for later json dump
                        reid_masks_anns.append({
                            'index': det_metadata.global_index,
                            'masks_path': str(abs_filepath),
                        })
                        pbar.update(1)

        print('Saving reid human parsing annotations as json to "{}"'.format(reid_anns_filepath))
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(reid_masks_anns).to_json(reid_anns_filepath)

    def load_reid_annotations(self, reid_anns_filepath, gt_dets):
        reid_anns = pd.read_json(reid_anns_filepath)
        assert len(reid_anns) == len(gt_dets), "reid_anns_filepath and gt_dets must have the same length"
        merged_df = pd.merge(gt_dets, reid_anns, left_index=False, right_index=False, left_on='global_index', right_on='index', validate='one_to_one')
        merged_df.drop('index', axis=1, inplace=True)
        return merged_df

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
            kpx, kpy = kpx * new_w / w , kpy * new_h / h

            rescaled_keypoints[i] = np.array([int(kpx), int(kpy), 1])
        return rescaled_keypoints, discarded_keypoints

    def gkern(self, kernlen=21, std=None):
        """Returns a 2D Gaussian kernel array."""
        if std is None:
            std = kernlen / 4
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def build_gaussian_heatmaps(self, kp_xyc, k, w, h, gaussian=None):
        gaussian_heatmaps = np.zeros((k, h, w))
        for i, kp in enumerate(kp_xyc):
            # do not use invisible keypoints
            if kp[2] == 0:
                continue

            kpx, kpy = kp[:2].astype(int)

            if gaussian is None:
                g_scale = 6
                g_radius = int(w / g_scale)
                gaussian = self.gkern(g_radius * 2 + 1)
            else:
                g_radius = gaussian.shape[0] // 2

            rt, rb = min(g_radius, kpy), min(g_radius, h - 1 - kpy)
            rl, rr = min(g_radius, kpx), min(g_radius, w - 1 - kpx)

            gaussian_heatmaps[i, kpy - rt:kpy + rb + 1, kpx - rl:kpx + rr + 1] = gaussian[
                                                                          g_radius - rt:g_radius + rb + 1,
                                                                          g_radius - rl:g_radius + rr + 1]
        return gaussian_heatmaps

    def filter_reid_samples(self, dets_df, max_total_ids=-1, min_vis=-1, min_h=0, min_w=0, min_samples=0, max_samples_per_id=10000):
        # filter detections by visibility
        dets_df_f1 = dets_df[dets_df['visibility'] >= min_vis]

        keep = (dets_df_f1['reid_crop_height'] >= min_h) & (dets_df_f1['reid_crop_width'] >= min_w)
        dets_df_f2 = dets_df_f1[keep]
        print("{} removed because too small samples (h<{} or w<{}) = {}".format(self.__class__.__name__, min_h, min_w, len(dets_df_f1) - len(dets_df_f2)))

        dets_df_f3 = dets_df_f2.groupby('person_id').apply(uniform_tracklet_sampling, max_samples_per_id, 'image_id').reset_index(drop=True).copy()
        print("{} removed for uniform tracklet sampling = {}".format(self.__class__.__name__, len(dets_df_f2) - len(dets_df_f3)))

        # Keep only ids with at least MIN_SAMPLES appearances
        count_per_id = dets_df_f3.person_id.value_counts()
        ids_to_keep = count_per_id.index[count_per_id.ge(min_samples)]
        dets_df_f4 = dets_df_f3[dets_df_f3.person_id.isin(ids_to_keep)]
        print("{} removed for not enough samples per id = {}".format(self.__class__.__name__, len(dets_df_f3) - len(dets_df_f4)))

        # Keep only max_total_ids ids
        if max_total_ids == -1:
            max_total_ids = len(dets_df_f4.person_id.unique())
        ids_to_keep = np.random.choice(dets_df_f4.person_id.unique(), replace=False, size=max_total_ids)
        dets_df_f5 = dets_df_f4[dets_df_f4.person_id.isin(ids_to_keep)]

        dets_df_f5.reset_index(drop=True, inplace=True)
        print("{} filtered size = {}".format(self.__class__.__name__, len(dets_df_f5)))

        return dets_df_f5

    def query_gallery_split(self, df, ratio):
        def uniform_tracklet_sampling(_df):
            x = list(_df.index)
            size = ceil(len(x) * ratio)
            result = list(np.random.choice(x, size=size, replace=False))
            return _df.loc[result]

        queries_per_pid = df.groupby('person_id').apply(uniform_tracklet_sampling)
        df.loc[df.id.isin(queries_per_pid.id), 'split'] = 'query'
        df.loc[~df.id.isin(queries_per_pid.id), 'split'] = 'gallery'
        return df

