from __future__ import absolute_import, division, print_function

import json
import os.path as osp
import cv2
import numpy as np
import pandas as pd

from os import path as osp
from pathlib import Path

import torch
from skimage.transform import resize
from tqdm import tqdm

from reconnaissance.utils.images_operations import overlay_heatmap
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
    df.rename(columns={'pid': 'pid_old'}, inplace=True)

    # Relabel Ids from 0 to N-1
    ids_df = df[['pid_old']].drop_duplicates()
    ids_df['pid'] = np.arange(ids_df.shape[0])
    df = df.merge(ids_df)
    return df


def to_dict_list(df):
    return df.to_dict('records')


def random_sampling_per_pid(df, ratio=1.0):
    def uniform_tracklet_sampling(_df):
        x = list(_df.unique())
        assert len(x) == len(_df)
        return list(np.random.choice(x, size=int(np.rint(len(x) * ratio)), replace=False))

    per_pid = df.groupby('pid')['index'].agg(uniform_tracklet_sampling)
    return per_pid.explode()


def clip_to_img_dim(bbox_ltwh, img_w, img_h):  # TODO put in utils
    l, t, w, h = bbox_ltwh.copy()
    l = max(l, 0)
    t = max(t, 0)
    w = min(l+w, img_w) - l
    h = min(t+h, img_h) - t
    return np.array([l, t, w, h])


class PoseTrack21ReID(ImageDataset):
    dataset_dir = 'Posetrack21'
    eval_metric = 'posetrack21_reid'
    annotations_dir = 'posetrack_data'
    img_ext = '.jpg'
    masks_ext = '.np'

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in PoseTrack21ReID.masks_dirs:
            return None
        else:
            return masks_dir[masks_dir]

        # TODO check if filtering is working
        # TODO filter FIRST: generate reid data only for filtered detections
        # TODO batch processing of heatmaps
        # TODO heatmaps from gaussians around KP
        # todo add nice prints
        # TODO add a occlusion score to each detection
    def __init__(self, config, datasets_root='', masks_dir='', crop_dim=(384, 128), pose_model=None, **kwargs):

        # Init
        datasets_root = Path(osp.abspath(osp.expanduser(datasets_root)))
        self.dataset_dir = Path(datasets_root, self.dataset_dir)
        self.train_anns_path = Path(self.dataset_dir, self.annotations_dir, 'train')
        self.val_anns_path = Path(self.dataset_dir, self.annotations_dir, 'val')
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
        self.load_reid_crops_annotations(train_reid_anns_filepath, self.train_gt_dets)
        self.load_reid_crops_annotations(val_reid_anns_filepath, self.val_gt_dets)

        # Build human parsing pseudo ground truth using the pose model
        existing_train_files = list(self.reid_val_path.glob('*/*{}'.format(self.masks_ext)))
        train_masks_anns_filepath = self.reid_mask_path / reid_anns_dir / ('val_' + masks_anns_filename)
        if len(existing_train_files) != len(self.train_gt_dets) or not train_masks_anns_filepath.exists():
            self.build_human_parsing_gt(self.reid_mask_path, self.reid_fig_path, 'train', train_masks_anns_filepath, self.train_gt_dets, self.train_images, crop_dim)


        nids = -1
        nsamples = -1
        max_samples_per_id = -1
        min_framerate = 4
        # virtual columns
        # config:
        # output size
        # filename format: "person-id_vid_img"
        # 2 of the three:
        # total number of samples
        # total number of ids
        # total number of sample per id
        # priority to:
        # occlusions
        # ids with multiple vid
        # id from different videos

        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None

        self.min_vis = config.data.motchallenge.min_vis
        self.min_h = config.data.motchallenge.min_h
        self.min_w = config.data.motchallenge.min_w
        self.min_samples_per_id = config.data.motchallenge.min_samples_per_id
        self.max_samples_per_id = config.data.motchallenge.max_samples_per_id
        self.train_ratio = config.data.motchallenge.train_ratio
        self.ratio_query_per_id = config.data.motchallenge.ratio_query_per_id
        self.ratio_gallery_per_id = config.data.motchallenge.ratio_gallery_per_id

        assert self.max_samples_per_id >= self.min_samples_per_id

        train_df = self.filter_reid_samples(self.train_gt_dets,
                                      self.min_vis,
                                      min_h=self.min_h,
                                      min_w=self.min_w,
                                      min_samples=self.min_samples_per_id,
                                      max_samples_per_id=self.max_samples_per_id)

        test_df = self.filter_reid_samples(self.val_gt_dets,
                                      min_samples=self.min_samples_per_id,
                                      max_samples_per_id=self.max_samples_per_id)

        train_df = relabel_ids(train_df)
        test_df = relabel_ids(test_df)
        query_df, gallery_df = self.split_query_gallery(test_df)

        train_df['camid'] = 0
        query_df['camid'] = 1
        gallery_df['camid'] = 2

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
        gt_dets = pd.concat(annotations_list).reset_index()
        categories = pd.concat(categories_list).reset_index()
        images = pd.concat(images_list).reset_index()
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
                    img_id = img_metadata.image_id
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
                            'index': det_metadata.index,
                            'reid_crop_path': str(abs_filepath),
                            'reid_crop_width': img_crop.shape[0],
                            'reid_crop_height': img_crop.shape[1],
                        })
                        pbar.update(1)
        print('Saving reid crops annotations as json to "{}"'.format(reid_anns_filepath))
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(reid_crops_anns).to_json(reid_anns_filepath)

    def build_human_parsing_gt(self, hp_save_path, fig_save_path, set_name, reid_anns_filepath, gt_dets_df, images_df, crop_dim):
        """
        Save on disk all human parsing gt for each reid crop.
        Create a json annotation file with human parsing metadata.
        """
        hp_save_path = hp_save_path / set_name
        fig_save_path = fig_save_path / set_name
        max_h, max_w = crop_dim
        pd.Series(dtype='int')
        reid_hp_anns = []
        with tqdm(total=len(gt_dets_df)) as pbar:
            pbar.set_description('Extracting all {} human parsing labels'.format(set_name))
            for vid_id in images_df.vid_id.unique():
                for img_metadata in images_df[images_df.vid_id == vid_id].itertuples():
                    filename = img_metadata.file_name
                    img_id = img_metadata.image_id
                    img_detections = gt_dets_df[gt_dets_df.image_id == img_id]
                    if len(img_detections) == 0:
                        continue
                    img = cv2.imread(str(self.dataset_dir / filename))
                    _, hp_gt_or = self.pose_model.run(torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0))
                    hp_gt_or = hp_gt_or.squeeze(0).permute((1, 2, 0)).numpy()
                    # cv2.imwrite("/Users/vladimirsomers/Downloads/test_hp_gt_1.jpg", overlay_heatmap(img, hp_gt_or.max(axis=2)))
                    hp_gt = resize(hp_gt_or, (img.shape[0], img.shape[1], hp_gt_or.shape[2]))
                    cv2.imwrite("/Users/vladimirsomers/Downloads/test_hp_gt.jpg", overlay_heatmap(img, hp_gt.max(axis=2)))
                    for det_metadata in img_detections.itertuples():
                        bbox = np.array(det_metadata.bbox)
                        bbox = clip_to_img_dim(bbox, img.shape[1], img.shape[0])
                        l, t, w, h = bbox.astype(int)
                        img_crop = img[t:t + h, l:l + w]
                        img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)
                        # hp_gt_crop from big hp_gt
                        hp_gt_crop = hp_gt[t:t + h, l:l + w]
                        # hp_gt_crop from small img_crop
                        # _, hp_gt_crop = self.pose_model.run(torch.from_numpy(img_crop).permute((2, 0, 1)).unsqueeze(0))
                        # hp_gt_crop = hp_gt_crop.squeeze().permute((1, 2, 0)).numpy()
                        hp_gt_crop = np.resize(hp_gt_crop, (max_h, max_w, hp_gt.shape[2]))
                        pid = det_metadata.person_id
                        filename = "{}_{}_{}".format(pid, vid_id, img_id)
                        abs_filepath = Path(hp_save_path, Path(vid_id, filename + self.masks_ext))
                        abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(abs_filepath), hp_gt)

                        img_with_heatmap = overlay_heatmap(img_crop, hp_gt_crop.max(axis=2))
                        figure_filepath = Path(fig_save_path, vid_id, filename + self.img_ext)
                        figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(figure_filepath), img_with_heatmap)

                        reid_hp_anns.append({
                            'index': det_metadata.index,
                            'reid_crop_path': str(abs_filepath),
                        })
                        pbar.update(1)

        print('Saving reid human parsing annotations as json to "{}"'.format(reid_anns_filepath))
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(reid_hp_anns).to_json(reid_anns_filepath)

    def filter_reid_samples(self, dets_df, min_vis=0, min_h=0, min_w=0, min_samples=0, max_samples_per_id=10000):
        dets_df_f1 = dets_df
        keep = (dets_df_f1['reid_crop_height'] >= min_h) & (dets_df_f1['reid_crop_width'] >= min_w)
        dets_df_f2 = dets_df_f1[keep]
        print("{} removed because too small samples (h<{} or w<{}) = {}".format(self.__class__.__name__, min_h, min_w, len(dets_df_f1) - len(dets_df_f2)))

        dets_df_f3 = dets_df_f2.groupby('person_id').apply(uniform_tracklet_sampling, max_samples_per_id, 'image_id').reset_index(drop=True).copy()
        print("{} removed for uniform tracklet sampling = {}".format(self.__class__.__name__, len(dets_df_f2) - len(dets_df_f3)))

        # Keep only ids with at least MIN_SAMPLES appearances
        dets_df_f3['samples_per_id'] = dets_df_f3.groupby('pid')['height'].transform('count').values
        dets_df_f4 = dets_df_f3[dets_df_f3['samples_per_id'] >= min_samples].copy()
        print("{} removed for not enough samples per id = {}".format(self.__class__.__name__, len(dets_df_f3) - len(dets_df_f4)))

        dets_df_f4.reset_index(inplace=True)
        dets_df_f4['index'] = dets_df_f4.index.values

        print("{} filtered size = {}".format(self.__class__.__name__, len(dets_df_f4)))

        return dets_df_f4

    def split_query_gallery(self, df):
        np.random.seed(0)
        query_per_id = random_sampling_per_pid(df, self.ratio_query_per_id)
        query_df = df.loc[query_per_id.values].copy()
        gallery_df = df.drop(query_per_id).copy()

        gallery_per_id = random_sampling_per_pid(gallery_df, self.ratio_gallery_per_id)
        gallery_df = gallery_df.loc[gallery_per_id.values].copy()

        return query_df, gallery_df

    def load_reid_crops_annotations(self, reid_anns_filepath, gt_dets):
        reid_anns = pd.read_json(reid_anns_filepath)
        return pd.concat([gt_dets, reid_anns], axis=1)
