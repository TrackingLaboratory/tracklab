import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from dd_sort.simformer.transforms import OfflineTransforms
from tracklab.engine.engine import merge_dataframes
from tracklab.utils.cv2 import cv2_load_image
from posetrack21_mot.motmetrics.distances import iou_matrix
from posetrack21_mot.motmetrics.lap import linear_sum_assignment


def add_crops(df, metadatas, **_):
    def get_crop(s):
        metadata = metadatas.loc[s.image_id]
        image = cv2_load_image(metadata.file_path)
        l, t, r, b = s.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        return image[t:b, l:r]

    df["bbox_conf"] = 1.0
    df["crop"] = df.apply(get_crop, axis=1)
    return df


def normalize2image(df, metadatas, **_):
    image_shape = cv2_load_image(metadatas.loc[df.iloc[0].image_id].file_path).shape[
                  :-1]  # FIXME add image shape in image metadata
    image_shape = image_shape[::-1]  # CV2 gives shapes in (H, W) format, but keypoints/bbox are in (W, H) format

    def norm_kps(kps):
        kps[:, :2] = kps[:, :2] / image_shape
        return kps

    df["bbox_ltwh"] = df["bbox_ltwh"].apply(lambda x: x / (list(image_shape) * 2))
    df["keypoints_xyc"] = df["keypoints_xyc"].apply(norm_kps)
    return df


class FakeEngine:
    num_workers = 8


def add_gt_reid_embeddings(df, metadatas, preds, tracker_state, pipeline, **_):
    """
        Add the appearance embedings from the loaded model to the dataframe based on the
        ground truth bboxes.
    """
    image_filepaths = metadatas['file_path'].to_dict()
    reider = next((mod for mod in pipeline if 'reid' in mod.__class__.__name__.lower()),
                  None)  # FIXME shouldn't be searched like this
    reider.datapipe.update(image_filepaths, metadatas, df)
    reids = []
    for idxs, batch in tqdm(reider.dataloader(engine=FakeEngine()),
                            desc=f"Embeddings using {reider.__class__.__name__}"):
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else list(idxs)
        batch_detections = df.loc[idxs]
        reids.append(reider.process(batch, batch_detections, metadatas))
    return merge_dataframes(df, reids)


def add_gt_poses(df, metadatas, preds, tracker_state, pipeline, **_):
    """
        Add the keypoints pose from the loaded model to the dataframe based on the
        ground truth bboxes.
    """
    image_filepaths = metadatas['file_path'].to_dict()
    pose_model = next((mod for mod in pipeline if 'pose' in mod.__class__.__name__.lower()),
                      None)  # FIXME shouldn't be searched like this
    pose_model.datapipe.update(image_filepaths, metadatas, df)
    poses = []
    for idxs, batch in tqdm(pose_model.dataloader(engine=FakeEngine()),
                            desc=f"Poses using {pose_model.__class__.__name__}"):
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else list(idxs)
        batch_detections = df.loc[idxs]
        poses.append(pose_model.process(batch, batch_detections, metadatas))
    return merge_dataframes(df, poses)


def add_last_obs_counter(df, metadatas, preds, tracker_state, **_):
    """
        Compute the number of frames since the last observation for each track.
    """
    df_sorted = df.sort_values(by=['track_id', 'image_id'])
    df_sorted['no_obs'] = 0

    for track_id in df_sorted['track_id'].unique():
        track_data = df_sorted[df_sorted['track_id'] == track_id]
        image_ids = track_data['image_id'].values

        diff = image_ids[1:] - image_ids[:-1]
        no_obs = [0]
        no_obs.extend(diff - 1)
        df_sorted.loc[track_data.index, 'no_obs'] = no_obs
    return df_sorted


def add_number_of_occlusions(df, metadatas, preds, tracker_state, **_):
    """
        Compute the number of occlusions for each track at each frame.
        Two detections are considered as occluded if IoU > 0.25.
    """
    df['occlusion_count'] = 0
    grouped = df.groupby('image_id')

    for image_id, group in grouped:
        bboxes = group['bbox_ltwh'].tolist()
        iou_matrix = compute_iou_matrix(bboxes)
        occlusion_count = (iou_matrix > 0.25).sum(axis=1)

        df.loc[group.index, 'occlusion_count'] = occlusion_count
    return df


def compute_iou_matrix(bboxes):
    bboxes = np.array(bboxes)

    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = x_min + bboxes[:, 2]
    y_max = y_min + bboxes[:, 3]

    areas = (x_max - x_min) * (y_max - y_min)

    num_boxes = len(bboxes)
    iou_matrix = np.zeros((num_boxes, num_boxes))

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            inter_x_min = max(x_min[i], x_min[j])
            inter_y_min = max(y_min[i], y_min[j])
            inter_x_max = min(x_max[i], x_max[j])
            inter_y_max = min(y_max[i], y_max[j])

            inter_width = max(0, inter_x_max - inter_x_min)
            inter_height = max(0, inter_y_max - inter_y_min)
            inter_area = inter_width * inter_height

            union_area = areas[i] + areas[j] - inter_area
            iou = inter_area / union_area

            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    return iou_matrix

def add_detections(df, metadatas, preds, tracker_state, **_):
    new_preds = []
    for image_id, image_df in df.groupby("image_id"):
        image_preds = preds.loc[preds.image_id == image_id].copy()
        disM = iou_matrix(np.array(image_df.bbox_ltwh.tolist()), np.array(image_preds.bbox_ltwh.tolist()),
                          max_iou=0.5)  # np.stack crashed if len() == 0
        le, ri = linear_sum_assignment(disM)

        image_preds["pred_track_id"] = image_preds["track_id"]
        image_preds["track_id"] = -1 * np.arange(len(image_preds)) - 1
        image_preds["person_id"] = -1 * np.arange(len(image_preds)) - 1
        image_preds["track_id"].iloc[ri] = image_df.track_id.iloc[le]
        image_preds["person_id"].iloc[ri] = image_df.person_id.iloc[le]
        new_preds.append(image_preds)
    new_df = merge_dataframes(pd.DataFrame(), new_preds)
    return new_df


def add_detections_with_id_switch(df, metadatas, preds, tracker_state, **_):
    tracklets = []
    new_preds = []
    old_dataframe = pd.DataFrame()
    ctc = {}  # current track correspondences
    for i, (image_id, image_df) in enumerate(df.groupby("image_id", sort=True)):
        image_preds = preds.loc[preds.image_id == image_id].copy()
        image_preds["id_switch"] = 0
        old_dataframe = merge_dataframes(old_dataframe, image_preds)
        disM = iou_matrix(np.array(image_df.bbox_ltwh.tolist()), np.array(image_preds.bbox_ltwh.tolist()), max_iou=0.5)
        le, ri = linear_sum_assignment(disM)
        for det_track, pred_idx in zip(image_df.track_id.iloc[le], image_preds.iloc[ri].index):
            pred_track = image_preds.track_id.at[pred_idx]
            if not np.isnan(pred_track):
                if det_track in ctc and ctc[det_track] != pred_track:
                    image_preds.at[pred_idx, "id_switch"] = 1
                ctc[det_track] = int(pred_track)
                # if i > 0:
                #     tracklets.append(old_dataframe.loc[old_dataframe.track_id==pred_track].index)
        image_preds["pred_track_id"] = image_preds["track_id"]
        image_preds["track_id"] = -1 * np.arange(len(image_preds)) - 1
        image_preds["person_id"] = -1 * np.arange(len(image_preds)) - 1
        image_preds["track_id"].iloc[ri] = image_df.track_id.iloc[le]
        image_preds["person_id"].iloc[ri] = image_df.person_id.iloc[le]
        new_preds.append(image_preds)
    new_df = merge_dataframes(pd.DataFrame(), new_preds)
    return new_df


OfflineTransforms.register("add_gt_poses", add_gt_poses)
OfflineTransforms.register("add_gt_reid_embeddings", add_gt_reid_embeddings)
OfflineTransforms.register("normalize2image", normalize2image)
OfflineTransforms.register("add_last_obs_counter", add_last_obs_counter)
OfflineTransforms.register("add_number_of_occlusions", add_number_of_occlusions)
OfflineTransforms.register("add_crops", add_crops)
OfflineTransforms.register("add_detections", add_detections)
OfflineTransforms.register("add_detections_with_id_switch", add_detections_with_id_switch)

INIT_STD = 0.1


def compute_oks_matrix(pred_dets, gt_dets):
    raise NotImplementedError()


def compute_sim_pose(det1, det2, aligned=False):
    kp1 = det1.keypoints_xyc
    kp2 = det2.keypoints_xyc

    vis_kp = np.logical_and(kp1[:, 2] == 1.0, kp2[:, 2] == 1.0)
    if vis_kp.sum() < 3:
        return 0.5

    kp1 = kp1[vis_kp, :2]
    kp2 = kp2[vis_kp, :2]
    # if aligned:
    #     kp2 = align_skeletons(kp2, kp1)

    scale = np.sqrt(
        (kp1[:, 0].max() - kp1[:, 0].min()) * (kp1[:, 1].max() - kp1[:, 1].min())
    )  # sqrt(area)
    scale = max(1.0, scale)  # avoid division by 0

    sigma = INIT_STD

    dist = np.linalg.norm(kp1 - kp2, axis=1)
    oks = np.exp(-0.5 * dist ** 2 / (scale ** 2 * sigma ** 2))
    oks = np.mean(oks)
    return oks
