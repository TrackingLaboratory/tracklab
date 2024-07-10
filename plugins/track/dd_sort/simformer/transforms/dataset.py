import numpy as np
import pandas as pd
import torch

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
        :-1
    ]  # FIXME add image shape in image metadata
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
    image_filepaths = metadatas['file_path'].to_dict()
    reider = next((mod for mod in pipeline if 'reid' in mod.__class__.__name__.lower()), None)  # FIXME shouldn't be searched like this
    reider.datapipe.update(image_filepaths, metadatas, df)
    reids = []
    for idxs, batch in reider.dataloader(engine=FakeEngine()):
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        batch_detections = df.loc[idxs]
        reids.append(reider.process(batch, batch_detections, metadatas))
    return merge_dataframes(df, reids)


def add_detections(df, metadatas, preds, tracker_state, **_):
    new_preds = []
    for image_id, image_df in df.groupby("image_id"):
        image_preds = preds.loc[preds.image_id == image_id].copy()
        disM = iou_matrix(np.array(image_df.bbox_ltwh.tolist()), np.array(image_preds.bbox_ltwh.tolist()), max_iou=0.5)  # np.stack crashed if len() == 0
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

OfflineTransforms.register("add_crops", add_crops)
OfflineTransforms.register("normalize2image", normalize2image)
OfflineTransforms.register("add_gt_reid_embeddings", add_gt_reid_embeddings)
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
    oks = np.exp(-0.5 * dist**2 / (scale**2 * sigma**2))
    oks = np.mean(oks)
    return oks
