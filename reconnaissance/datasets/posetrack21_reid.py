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

from reconnaissance.utils.images import overlay_heatmap
from torchreid.data import ImageDataset


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


def relabel_ids(df):
    # create pids as 0-based increasing numbers
    df['pid'] = pd.factorize(df.person_id)[0]
    return df


def to_dict_list(df):
    return df.to_dict('records')


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


def clip_to_img_dim(bbox_ltwh, img_w, img_h):  # TODO put in utils
    l, t, w, h = bbox_ltwh.copy()
    l = max(l, 0)
    t = max(t, 0)
    w = min(l+w, img_w) - l
    h = min(t+h, img_h) - t
    return np.array([l, t, w, h])


class PoseTrack21ReID(ImageDataset):
    dataset_dir = 'Posetrack21'
    # eval_metric = 'posetrack21_reid'
    eval_metric = 'motchallenge'
    annotations_dir = 'posetrack_data'
    img_ext = '.jpg'
    masks_ext = '.npy'

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

        # TODO run it on Belldev
        # ----
        # TODO refactor filter config
        # TODO filter FIRST: generate reid data only for filtered detections
        # TODO keeps person_id and use pid as index from 0
        # TODO add config to tell if identities holds across videos or not
        # TODO create general purpose eval metric for MOT datasets
        # TODO fix all todos in code
        # TODO bootstrap general purpose dataset class for MOT datasets (will be improved later when integrating with other datasets)
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


    def __init__(self, config, datasets_root='', masks_dir='', crop_dim=(384, 128), pose_model=None, **kwargs):

        # Init
        datasets_root = Path(osp.abspath(osp.expanduser(datasets_root)))
        self.dataset_dir = Path(datasets_root, self.dataset_dir)
        self.train_anns_path = Path(self.dataset_dir, self.annotations_dir, 'train')
        self.val_anns_path = Path(self.dataset_dir, self.annotations_dir, 'val')
        self.cfg = config.data.mot
        assert self.cfg.max_samples_per_id >= self.cfg.min_samples_per_id, "max_samples_per_id must be >= min_samples_per_id"

        reid_dir = 'reid'
        reid_images_dir = 'images'
        reid_masks_dir = 'masks'
        reid_fig_dir = 'figures'
        reid_anns_dir = 'anns'
        images_anns_filename = 'images_anns.json'
        masks_anns_filename = 'masks_anns.json'

        # TODO remove 'self.'
        self.reid_path = Path(self.dataset_dir, reid_dir)
        self.reid_img_path = self.reid_path / reid_images_dir
        self.reid_mask_path = self.reid_path / reid_masks_dir
        self.reid_mask_train_path = self.reid_mask_path / 'train'
        self.reid_mask_val_path = self.reid_mask_path / 'val'
        self.reid_fig_path = self.reid_path / reid_fig_dir
        self.reid_train_path = self.reid_img_path / 'train'
        self.reid_val_path = self.reid_img_path / 'val'
        self.pose_model = pose_model

        # Load annotations into Pandas dataframes
        self.train_categories, self.train_gt_dets, self.train_images = self.build_annotations_df(self.train_anns_path)
        self.val_categories, self.val_gt_dets, self.val_images = self.build_annotations_df(self.val_anns_path)

        # Save detections crops and related metadata on disk to build ReID dataset
        existing_train_files = list(self.reid_train_path.glob('*/*{}'.format(self.img_ext)))
        train_reid_anns_filepath = self.reid_img_path / reid_anns_dir / ('train_' + images_anns_filename)
        if len(existing_train_files) != len(self.train_gt_dets) or not train_reid_anns_filepath.exists():
            self.build_reid_set(self.reid_img_path, 'train', train_reid_anns_filepath, self.train_gt_dets, self.train_images, crop_dim)

        existing_val_files = list(self.reid_val_path.glob('*/*{}'.format(self.img_ext)))
        val_reid_anns_filepath = self.reid_img_path / reid_anns_dir / ('val_' + images_anns_filename)
        if len(existing_val_files) != len(self.val_gt_dets) or not val_reid_anns_filepath.exists():
            self.build_reid_set(self.reid_img_path, 'val', val_reid_anns_filepath, self.val_gt_dets, self.val_images, crop_dim)

        # Load reid crops metadata into existing ground truth detections dataframe
        self.train_gt_dets = self.load_reid_annotations(train_reid_anns_filepath, self.train_gt_dets)
        self.val_gt_dets = self.load_reid_annotations(val_reid_anns_filepath, self.val_gt_dets)

        # Build human parsing pseudo ground truth using the pose model
        existing_train_files = list(self.reid_mask_train_path.glob('*/*{}'.format(self.masks_ext)))
        train_masks_anns_filepath = self.reid_mask_path / reid_anns_dir / ('train_' + masks_anns_filename)
        if len(existing_train_files) != len(self.train_gt_dets) or not train_masks_anns_filepath.exists():
            self.build_human_parsing_gt(self.reid_mask_path, self.reid_fig_path, 'train', train_masks_anns_filepath, self.train_gt_dets, self.train_images, (128, 64), (32, 16))

        existing_val_files = list(self.reid_mask_val_path.glob('*/*{}'.format(self.masks_ext)))
        val_masks_anns_filepath = self.reid_mask_path / reid_anns_dir / ('val_' + masks_anns_filename)
        if len(existing_val_files) != len(self.val_gt_dets) or not val_masks_anns_filepath.exists():
            self.build_human_parsing_gt(self.reid_mask_path, self.reid_fig_path, 'val', val_masks_anns_filepath, self.val_gt_dets, self.val_images, (128, 64), (32, 16))

        # Load reid masks metadata into existing ground truth detections dataframe
        self.train_gt_dets = self.load_reid_annotations(train_masks_anns_filepath, self.train_gt_dets)
        self.val_gt_dets = self.load_reid_annotations(val_masks_anns_filepath, self.val_gt_dets)

        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix, self.masks_parts_names = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix, self.masks_parts_names = None, None, None, None

        train_df = self.filter_reid_samples(self.train_gt_dets,
                                            max_ids=self.cfg.max_ids,
                                            min_vis=self.cfg.min_vis,
                                            min_h=self.cfg.min_h,
                                            min_w=self.cfg.min_w,
                                            min_samples=self.cfg.min_samples_per_id,
                                            max_samples_per_id=self.cfg.max_samples_per_id)

        test_df = self.filter_reid_samples(self.val_gt_dets,
                                           max_ids=self.cfg.max_ids,
                                           min_samples=self.cfg.min_samples_per_id,
                                           max_samples_per_id=self.cfg.max_samples_per_id)

        # FIXME mark detetions as query/gallery with a new columns instead of using a new dataframe
        # -> that will also fix warning
        train_df = relabel_ids(train_df)
        test_df = relabel_ids(test_df)
        query_df, gallery_df = self.split_query_gallery(test_df, self.ratio_query_per_id)

        train_df['camid'] = 0 # TODO put video id + explain why
        query_df['camid'] = 1
        gallery_df['camid'] = 2

        # train_df.rename(columns={'person_id': 'pid'}, inplace=True)
        # query_df.rename(columns={'person_id': 'pid'}, inplace=True)
        # gallery_df.rename(columns={'person_id': 'pid'}, inplace=True)

        train_df.rename(columns={'reid_crop_path': 'img_path'}, inplace=True)
        query_df.rename(columns={'reid_crop_path': 'img_path'}, inplace=True)
        gallery_df.rename(columns={'reid_crop_path': 'img_path'}, inplace=True)

        train_df.drop(columns='bbox_head', inplace=True)
        query_df.drop(columns='bbox_head', inplace=True)
        gallery_df.drop(columns='bbox_head', inplace=True)

        train = to_dict_list(train_df)
        query = to_dict_list(query_df)
        gallery = to_dict_list(gallery_df)

        super(PoseTrack21ReID, self).__init__(train, query, gallery, **kwargs)

    def build_annotations_df(self, anns_path):
        annotations_list = []
        categories_list = []
        images_list = []
        for path in anns_path.glob('*.json'):
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

        def reshape_keypoints(keypoints):
            return np.array(keypoints).reshape(-1, 3)

        gt_dets.keypoints = gt_dets.keypoints.apply(reshape_keypoints)
        # compute detection visiblity as average keypoints visibility
        gt_dets['visibility'] = gt_dets.keypoints.apply(
            lambda x: x[:, 2].mean()
        )

        gt_dets['global_index'] = gt_dets.index

        # TODO :
        # - bbox: bbox_tlwh, bbox_tlbr, cycxhw,
        # - keypoints: keypoints_xyc, keypoints_bbox_xyc
        # function to compute keypints in rescaled bbox
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
            for vid_id in images_df.vid_id.unique():
                for img_metadata in images_df[images_df.vid_id == vid_id].itertuples():
                    filename = img_metadata.file_name
                    img_id = img_metadata.image_id  # TODO skip non labeled images
                    img = cv2.imread(str(self.dataset_dir / filename))
                    for det_metadata in gt_dets_df[gt_dets_df.image_id == img_id].itertuples():
                        bbox = np.array(det_metadata.bbox)
                        bbox = clip_to_img_dim(bbox, img.shape[1], img.shape[0])
                        pid = det_metadata.person_id
                        l, t, w, h = bbox.astype(int)
                        img_crop = img[t:t + h, l:l + w]
                        if h > max_h or w > max_w:
                            img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)
                        filename = "{}_{}_{}{}".format(pid, vid_id, img_id, self.img_ext)
                        rel_filepath = Path(vid_id, filename)
                        abs_filepath = Path(save_path, rel_filepath)
                        abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(abs_filepath), img_crop)
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

    def build_human_parsing_gt(self, hp_save_path, fig_save_path, set_name, reid_anns_filepath, gt_dets_df, images_df, crop_dim, masks_dim):
        """
        Save on disk all human parsing gt for each reid crop.
        Create a json annotation file with human parsing metadata.
        """
        hp_save_path = hp_save_path / set_name
        fig_save_path = fig_save_path / set_name
        crop_h, crop_w = crop_dim
        mask_h, mask_w = masks_dim
        total_discarded_keypoints = 0
        reid_hp_anns = []
        from_gaussian = True
        from_model_on_crops = False
        from_model_on_img = False
        g_scale = 6
        g_radius = int(mask_w / g_scale)
        gaussian = self.gkern(g_radius * 2 + 1)
        with tqdm(total=len(gt_dets_df)) as pbar:
            pbar.set_description('Extracting all {} human parsing labels'.format(set_name))
            for vid_id in images_df.vid_id.unique():
                img_shape = None
                for img_metadata in images_df[images_df.vid_id == vid_id].itertuples():
                    filename = img_metadata.file_name
                    img_id = img_metadata.image_id
                    img_detections = gt_dets_df[gt_dets_df.image_id == img_id]
                    if len(img_detections) == 0:
                        continue
                    if img_shape is None:
                        img_shape = cv2.imread(str(self.dataset_dir / filename)).shape  # Image loaded once to get video frame size
                    if from_model_on_img:
                        img = cv2.imread(str(self.dataset_dir / filename))
                        _, hp_gt_or = self.pose_model.run(torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0))  # TODO check if pose_model need BRG or RGB
                        hp_gt_or = hp_gt_or.squeeze(0).permute((1, 2, 0)).numpy()  # TODO why that permute needed? for old resize?
                        # cv2.imwrite("/Users/vladimirsomers/Downloads/test_hp_gt_1.jpg", overlay_heatmap(img, hp_gt_or.max(axis=2)))
                        hp_gt = resize(hp_gt_or, (img.shape[0], img.shape[1], hp_gt_or.shape[2]))
                        cv2.imwrite("/Users/vladimirsomers/Downloads/test_hp_gt.jpg", overlay_heatmap(img, hp_gt.max(axis=2)))  # FIXME
                    for det_metadata in img_detections.itertuples():
                        bbox = clip_to_img_dim(det_metadata.bbox, img_shape[1], img_shape[0]).astype(int)
                        keypoints = det_metadata.keypoints
                        l, t, w, h = bbox
                        if from_gaussian:
                            img_crop = cv2.imread(det_metadata.reid_crop_path)
                            img_crop = cv2.resize(img_crop, (crop_w, crop_h), cv2.INTER_CUBIC)
                            rf_keypoints, discarded_keypoints = self.rescale_and_filter_keypoints(keypoints, bbox, mask_w, mask_h)
                            hp_gt_crop = self.build_gaussian_heatmaps(rf_keypoints, len(keypoints), mask_w, mask_h, gaussian=gaussian)
                            total_discarded_keypoints += discarded_keypoints
                        elif from_model_on_crops:
                            img_crop = cv2.imread(det_metadata.reid_crop_path)
                            img_crop = cv2.resize(img_crop, (crop_w, crop_h), cv2.INTER_CUBIC)
                            _, hp_gt_crop = self.pose_model.run(torch.from_numpy(img_crop).permute((2, 0, 1)).unsqueeze(0))
                            hp_gt_crop = hp_gt_crop.squeeze().permute((1, 2, 0)).numpy()
                            hp_gt_crop = np.resize(hp_gt_crop, (crop_h, crop_w, hp_gt_crop.shape[2]))  # FIXME other resize
                        elif from_model_on_img:
                            img_crop = img[t:t + h, l:l + w]
                            img_crop = cv2.resize(img_crop, (crop_w, crop_h), cv2.INTER_CUBIC)
                            hp_gt_crop = hp_gt[t:t + h, l:l + w]
                            hp_gt_crop = np.resize(hp_gt_crop, (crop_h, crop_w, hp_gt_crop.shape[2]))  # FIXME other resize
                        else:
                            raise ValueError('Invalid human parsing method')

                        pid = det_metadata.person_id
                        filename = "{}_{}_{}".format(pid, vid_id, img_id)
                        abs_filepath = Path(hp_save_path, Path(vid_id, filename + self.masks_ext))
                        abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(abs_filepath), hp_gt_crop)

                        img_with_heatmap = overlay_heatmap(img_crop, hp_gt_crop.max(axis=0), weight=0.3)
                        figure_filepath = Path(fig_save_path, vid_id, filename + self.img_ext)
                        figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(figure_filepath), img_with_heatmap)

                        reid_hp_anns.append({
                            'index': det_metadata.global_index,
                            'masks_path': str(abs_filepath),
                        })
                        pbar.update(1)

        print("Discarded {} keypoints because out of bbox".format(total_discarded_keypoints))
        print('Saving reid human parsing annotations as json to "{}"'.format(reid_anns_filepath))
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(reid_hp_anns).to_json(reid_anns_filepath)

    def load_reid_annotations(self, reid_anns_filepath, gt_dets):
        reid_anns = pd.read_json(reid_anns_filepath)
        assert len(reid_anns) == len(gt_dets), "reid_anns_filepath and gt_dets must have the same length"
        merged_df = pd.merge(gt_dets, reid_anns, left_index=False, right_index=False, left_on='global_index', right_on='index', validate='one_to_one')
        merged_df.drop('index', axis=1, inplace=True)
        return merged_df

    def rescale_and_filter_keypoints(self, keypoints, bbox, new_w, new_h):
        l, t, w, h = bbox.astype(int)
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

    def build_gaussian_heatmaps(self, keypoints, k, w, h, gaussian=None):
        gaussian_heatmaps = np.zeros((k, h, w))
        for i, kp in keypoints.items():
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

    def filter_reid_samples(self, dets_df, max_ids=-1, min_vis=-1, min_h=0, min_w=0, min_samples=0, max_samples_per_id=10000):
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

        ids_to_keep = np.random.choice(dets_df_f4.person_id.unique(), replace=False, size=max_ids)
        dets_df_f5 = dets_df_f4[dets_df_f4.person_id.isin(ids_to_keep)]

        dets_df_f5.reset_index(drop=True, inplace=True)
        print("{} filtered size = {}".format(self.__class__.__name__, len(dets_df_f5)))

        return dets_df_f5

    def split_query_gallery(self, df, ratio):
        def uniform_tracklet_sampling(_df):
            x = list(_df.index)
            size = ceil(len(x) * ratio)
            result = list(np.random.choice(x, size=size, replace=False))
            return _df.loc[result]

        per_pid = df.groupby('person_id').apply(uniform_tracklet_sampling)
        query_df = df[df.id.isin(per_pid.id)]
        gallery_df = df[~df.id.isin(per_pid.id)]
        return query_df, gallery_df
