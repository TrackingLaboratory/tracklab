import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

from pbtrack.datastruct import TrackingDataset, TrackingSet
from posetrack21_mot.motmetrics.distances import iou_matrix
from posetrack21_mot.motmetrics.lap import linear_sum_assignment


class DanceTrack(TrackingDataset):

    def __init__(self, dataset_path: str, *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), f"'{self.dataset_path}' directory does not exist"
        self.annotation_path = self.dataset_path / "annotations"
        assert self.annotation_path.exists(), f"'{self.annotation_path}' directory does not exist"

        train_set = load_tracking_set(self.annotation_path, self.dataset_path, "train")
        val_set = load_tracking_set(self.annotation_path, self.dataset_path, "val")
        test_set = load_tracking_set(self.annotation_path, self.dataset_path, "test")
        super().__init__(dataset_path, train_set, val_set, test_set, *args, **kwargs)


def load_tracking_set(anns_path, dataset_path, split):
    with open(anns_path / f"{split}.json") as f:
        data_dict = json.load(f)
    image_metadatas = pd.DataFrame(data_dict["images"])
    image_metadatas.drop(["prev_image_id", "next_image_id"], axis=1, inplace=True)
    image_metadatas.rename({"file_name": "file_path", "frame_id": "frame", }, inplace=True, axis=1)
    image_metadatas["file_path"] = image_metadatas["file_path"].apply(lambda x: os.path.join(dataset_path, split, x))
    image_metadatas.set_index("id", drop=True, inplace=True)
    if split in ["train", "val"]:
        image_metadatas["is_labeled"] = True
    else:
        image_metadatas["is_labeled"] = False
    # fixme add ignore_regions_x and y = [] ?
    # fixme add is_labeled and/or has_labeled_person ?

    detections_gt = pd.DataFrame(data_dict["annotations"])
    if len(detections_gt):
        detections_gt.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
        detections_gt["bbox_ltwh"] = detections_gt.bbox_ltwh.apply(
            lambda x: np.array(x))
        detections_gt.drop(["conf", "iscrowd", "area"], axis=1, inplace=True)
        detections_gt.set_index("id", drop=True, inplace=True)
        detections_gt = detections_gt.merge(image_metadatas[["video_id"]], how="left", left_on="image_id",
            right_index=True)
        detections_gt["person_id"] = 1000 * detections_gt["video_id"] + detections_gt[
            "track_id"]
        if "keypoints_xyc" not in detections_gt:
            detections_gt = load_keypoints_from_centernet(dataset_path, split, detections_gt)
            data_dict["annotations"] = [{**x, "keypoints_xyc": detections_gt["keypoints_xyc"].loc[x["id"]].tolist()} for x in data_dict["annotations"]]
            with open(anns_path / f"{split}.json", "w") as f:
                json.dump(data_dict, f)
        else:
            detections_gt["keypoints_xyc"] = detections_gt.keypoints_xyc.apply(
                lambda x: np.array(x)
            )
    video_metadatas = pd.DataFrame(data_dict["videos"])
    image_count_per_video = image_metadatas.groupby('video_id').size().reset_index(name='nframes')
    video_metadatas = video_metadatas.merge(image_count_per_video, left_on="id", right_on="video_id", how="left")
    video_metadatas.drop(["video_id"], axis=1, inplace=True)
    video_metadatas.rename(columns={"file_name": "name"}, inplace=True)
    video_metadatas["categories"] = video_metadatas.apply(lambda x: data_dict["categories"], axis=1)
    video_metadatas.set_index("id", drop=True, inplace=True)

    return TrackingSet(video_metadatas, image_metadatas, detections_gt, split=split)


def load_keypoints_from_centernet(dataset_path, split, detections_gt):
    centernet_path = dataset_path / "centernet_dets"
    centernet_json_path = centernet_path / f"{split}_w_ttda.json"
    with centernet_json_path.open() as fp:
        data_dict = json.load(fp)
    centernet_dets = pd.DataFrame(data_dict["annotations"])
    detections_gt_kp = []
    for image_id, image_df in track(detections_gt.groupby("image_id"), description=f"Keypoints for {split}..."):
        image_preds = centernet_dets.loc[centernet_dets.image_id == image_id]
        disM = iou_matrix(np.array(image_df.bbox_ltwh.tolist()), np.array(image_preds["bbox"].tolist()), max_iou=0.5)
        le, ri = linear_sum_assignment(disM)
        image_df["keypoints_xyc"] = None
        image_df["keypoints_xyc"].iloc[le] = image_preds["keypoints"].iloc[ri]
        image_df["keypoints_xyc"] = image_df["keypoints_xyc"].apply(
            lambda x: np.array(x).reshape(17,3) if x else np.zeros((17, 3))
        )
        detections_gt_kp.append(image_df)

    return pd.concat(detections_gt_kp)


def load_annotations(anns_path):
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, "No annotations files found in {}".format(anns_path)
    detections_gt = []
    image_metadatas = []
    video_metadatas = []
    for path in anns_files_list:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            detections_gt.extend(data_dict["annotations"])
            image_metadatas.extend(data_dict["images"])
            video_metadatas.append(
                {"id": data_dict["images"][0]["vid_id"], "nframes": len(data_dict["images"]), "name": path.stem,
                    "categories": data_dict["categories"], })

    return (pd.DataFrame(video_metadatas), pd.DataFrame(image_metadatas), pd.DataFrame(detections_gt),)


def fix_formatting(video_metadatas, image_metadatas, detections_gt, dataset_path, posetrack_version):
    image_id = "image_id" if posetrack_version == 21 else "frame_id"

    # Videos
    video_metadatas.set_index("id", drop=True, inplace=True)

    # Images
    image_metadatas.drop([image_id, "nframes"], axis=1, inplace=True)
    image_metadatas["file_name"] = image_metadatas["file_name"].apply(lambda x: os.path.join(dataset_path, x))
    image_metadatas["frame"] = image_metadatas["file_name"].apply(lambda x: int(os.path.basename(x).split(".")[0]) + 1)
    image_metadatas.rename(columns={"vid_id": "video_id", "file_name": "file_path"}, inplace=True, )
    image_metadatas.set_index("id", drop=True, inplace=True)

    # Detections
    detections_gt.drop(["bbox_head"], axis=1, inplace=True)
    detections_gt.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
    detections_gt.bbox_ltwh = detections_gt.bbox_ltwh.apply(lambda x: np.array(x))
    detections_gt.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
    detections_gt.keypoints_xyc = detections_gt.keypoints_xyc.apply(lambda x: np.reshape(np.array(x), (-1, 3)))
    detections_gt.set_index("id", drop=True, inplace=True)
    detections_gt = detections_gt.merge(image_metadatas[["video_id"]], how="left", left_on="image_id", right_index=True)

    return video_metadatas, image_metadatas, detections_gt
