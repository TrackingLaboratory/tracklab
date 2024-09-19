import logging
from collections import defaultdict
from itertools import accumulate

import cv2
import pytorch_lightning as pl
import numpy as np
import torch

from typing import Optional, Any
from PIL import Image
from matplotlib import patches, pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pathlib import Path

from tracklab.utils import coordinates
from tracklab.utils.cv2 import cv2_load_image, draw_text

log = logging.getLogger(__name__)


class VisualizeTrackletBatches(pl.Callback):
    def __init__(self, tracking_sets, enabled_steps=None, max_batch_size=None, max_frames=None):
        self.tracking_sets = tracking_sets
        self.enabled_steps = enabled_steps
        self.epoch = 0
        self.batch_size = max_batch_size
        self.max_frames = max_frames

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch += 1

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        step = "train"
        if step not in self.enabled_steps:
            return
        self.display_batch(batch, outputs, step, batch_idx)

    def display_batch(self, batch, outputs, step, batch_idx) -> None:
        output_images = []
        batch_size = self.batch_size or len(batch["image_id"])
        for sample_idx in range(batch_size):
            image_id = batch['image_id'][sample_idx, 0, 0].cpu().numpy()
            track_image_paths = []
            track_image_paths_dict = {}
            if image_id == -1:
                continue
            image_path = self.tracking_sets[step].image_metadatas.loc[
                image_id].file_path
            for track_idx in range(len(batch["image_id"][sample_idx])):
                track_image_paths.append([])
                for det_idx in range(
                        len(batch["track_feats"]["image_id"][sample_idx, track_idx])):
                    img_id = batch["track_feats"]["image_id"][
                        sample_idx, track_idx, det_idx, 0].cpu().numpy()
                    if np.isnan(img_id):
                        continue
                    img_path = self.tracking_sets[step].image_metadatas.loc[
                        img_id].file_path
                    track_image_paths[track_idx].append(img_path)
                    track_image_paths_dict[track_idx, det_idx] = img_path

            output_image = self.display_sample(batch, sample_idx, image_path,
                                               track_image_paths,
                                               track_image_paths_dict)
            output_images.append(output_image)

        max_height = max([i.shape[0] for i in output_images])
        fig, axs = plt.subplots(ncols=len(output_images),
                                # figsize=(int(max_height/1080)*5, 3*len(output_images)),
                                figsize=(
                                6 * len(output_images), int(max_height / 1080) * 5),
                                # sharey=True,
                                squeeze=False,
                                layout="constrained")
        axs = axs.flatten()
        for image, ax in zip(output_images, axs):
            ax.imshow(image)
            ax.set_anchor("N")
        save_path = Path(f"visualization/{step}_epoch{self.epoch}_{batch_idx}.pdf")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close()

    def display_sample(self, batch, sample_idx, image_path, track_image_paths,
                       track_image_paths_dict):
        target_image = cv2_load_image(image_path)
        track_images_dict = {}
        image_per_age = {}
        dets_per_age = defaultdict(list)
        for k, track_image_path in track_image_paths_dict.items():
            age = int(batch["track_feats"]["age"][sample_idx, k[0], k[1]])
            if age not in image_per_age:
                image_per_age[age] = cv2_load_image(track_image_path)
            track_images_dict[k] = image_per_age[
                age]  # cv2_load_image(track_image_path)
            dets_per_age[age].append(k)

        track_ids = []
        ages = sorted(image_per_age.keys(), reverse=False)
        num_images = len(ages)
        cols = 4
        rows = int(np.ceil(num_images / cols))
        grid_size = int(np.ceil(np.sqrt(num_images)))
        img_h = target_image.shape[0]
        self.track_ids = []
        self.tracklet_lines = defaultdict(list)
        self.tracklet_colors = {}
        output_images = []

        target_image = self.draw_frame(target_image,
                                       [(i, 0) for i in
                                        range(len(batch["det_targets"][sample_idx])) if
                                        not np.isnan(
                                            batch["det_targets"][sample_idx, i].cpu())],
                                       batch["det_targets"][sample_idx],
                                       batch["det_feats"]["bbox_ltwh"][sample_idx],
                                       age=0,
                                       index=len(ages)
                                       )
        output_images.append(target_image)

        for i, age in enumerate(ages):
            if self.max_frames is not None and i > self.max_frames:
                break
            image = image_per_age[age]
            out_image = self.draw_frame(image, dets_per_age[age],
                                        batch["track_targets"][sample_idx],
                                        batch["track_feats"]["bbox_ltwh"][sample_idx],
                                        age, i)
            output_images.append(image)


        output_image = np.concatenate(output_images, axis=0)
        # for track_id, tracklet_line in self.tracklet_lines.items():
        #     color = self.tracklet_colors[track_id]
        #     tracklet_line = np.array(tracklet_line)
        #     cv2.polylines(output_image, [tracklet_line], isClosed=False, color=color, thickness=2)
        return output_image

    def draw_frame(self, image, dets, track_ids, bboxes, age, index):
        for det_track, det_idx in dets:
            track_id = int(track_ids[
                               det_track, det_idx])  # batch["track_targets"][sample_idx, det_track, det_idx])
            if track_id not in self.track_ids:
                self.track_ids.append(track_id)
                self.tracklet_colors[track_id] = np.array(
                    plt.cm.tab10(self.track_ids.index(track_id))) * 255
            color = self.tracklet_colors[
                track_id]  # np.array(plt.cm.tab10(track_ids.index(track_id))) * 255
            det = bboxes[
                det_track, det_idx].cpu().numpy()  # batch["track_feats"]["bbox_ltwh"][sample_idx, det_track, det_idx]
            ltrb = coordinates.bbox_ltwh2ltrb(det * np.array(
                [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))
            l, t, r, b = [int(x) for x in ltrb]
            center = (np.mean((l, r)).astype(int), (np.mean((t, b)).astype(int) + (
                        index * image.shape[
                    0])))  # find the center of the bbox modified by image position
            self.tracklet_lines[track_id].append(center)
            cv2.rectangle(image, (l, t), (r, b), color=color, thickness=3)
            draw_text(
                image,
                f"ID: {track_id}",
                (r - 5, t - 15),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                thickness=1,
                alignH="r",
                alignV="t",
                color_txt=color,
                color_bg=(255, 255, 255),
                alpha_bg=0.5,
            )
        draw_text(image,
                  f"AGE: {age}",
                  (20, 20),
                  fontFace=cv2.FONT_HERSHEY_DUPLEX,
                  fontScale=1,
                  thickness=1,
                  alignH="l",
                  alignV="t",
                  color_txt=(0, 0, 0),
                  color_bg=(255, 255, 255),
                  alpha_bg=0.5,
                  )
        return image

class DisplayBatchSamples(pl.Callback):
    def __init__(self, tracking_sets, plot=False, enabled_steps=None):
        self.positive_sim = []
        self.negative_sim = []
        self.tracking_sets = tracking_sets
        self.plot = plot
        self.enabled_steps = enabled_steps  # {'train', 'val', 'predict'}

    def on_train_batch_end(self,
                           trainer: "pl.Trainer",
                           pl_module: "pl.LightningModule",
                           outputs: Optional[STEP_OUTPUT],
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int = 0,):
        step = 'train'
        if step not in self.enabled_steps:
            return
        self.display_batch(batch, outputs, step)

    def on_predict_batch_end(self,
                             trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule",
                             outputs: Optional[STEP_OUTPUT],
                             batch: Any,
                             batch_idx: int,
                             dataloader_idx: int = 0,):
        if 'test' not in self.enabled_steps:
            return
        pass
        # TODO
        # loss, td_sim_matrix, preds, targets, pred_assoc_matrices, gt_assoc_matrices = outputs
        # det_feats = batch["det_feats"]
        # track_feats = batch["track_feats"]
        # det_keypoints_xyc = self.motionbert_preprocess(det_feats["keypoints_xyc"])
        # det_keypoints_xyc[..., :2] = det_keypoints_xyc[..., :2]/2 + 0.5
        # track_keypoints_xyc = self.motionbert_preprocess(track_feats["keypoints_xyc"])
        # track_keypoints_xyc[..., :2] = track_keypoints_xyc[..., :2]/2 + 0.5
        # display_bboxes(det_feats["bbox_ltwh"].squeeze(1).unsqueeze(0), det_masks, det_targets,
        #                track_feats["bbox_ltwh"][:, :1].squeeze(1).unsqueeze(0), track_masks, track_targets,
        #                det_keypoints_xyc.flatten(2, -1).squeeze(1).unsqueeze(0),
        #                track_keypoints_xyc.flatten(2, -1)[:, :1].squeeze(1).unsqueeze(0), None, batch["images"], posetrack=False)
        # # display_bboxes(det_feats["bbox_ltwh"].squeeze(1).unsqueeze(0), det_masks, det_targets,
        # #                track_feats["bbox_ltwh"][:, :1].squeeze(1).unsqueeze(0), track_masks, track_targets,
        # #                det_feats["keypoints_xyc"].squeeze(1).unsqueeze(0),
        # #                track_feats["keypoints_xyc"][:, :1].squeeze(1).unsqueeze(0), batch["images"])

    def on_validation_batch_end(self,
                                trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule",
                                outputs: Optional[STEP_OUTPUT],
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0,):
        step = 'val'
        if step not in self.enabled_steps:
            return
        self.display_batch(batch, outputs, step)

    def display_batch(self, batch, outputs, step):
        image_ids = batch['image_id'][:, 0, 0].cpu().numpy()
        image_paths = []
        for img_id in image_ids:
            img_path = self.tracking_sets[step].image_metadatas.loc[img_id].file_path
            image_paths.append(img_path)
        plt = display_bboxes(outputs["dets"], outputs["tracks"], image_paths, None)
        if self.plot:
            plt.show()


def display_bboxes(tracks, dets, image_paths, img_tensor, posetrack=True):

    # Move tensors to CPU
    det_bbox_ltwh = dets.feats['bbox_ltwh'].cpu()
    det_masks = dets.masks.cpu()
    if dets.targets is None:
        det_targets = torch.range(start=0, end=det_bbox_ltwh.shape[1]-1).unsqueeze(0)
    else:
        det_targets = dets.targets.cpu()
        det_targets = det_targets.nan_to_num(-1)
    track_bbox_ltwh = tracks.feats['bbox_ltwh'].cpu()
    track_masks = tracks.masks.cpu()
    if tracks.targets is None:
        track_targets = torch.range(start=0, end=track_bbox_ltwh.shape[1]-1).unsqueeze(0)
    else:
        track_targets = tracks.targets.cpu()
        track_targets = track_targets.nan_to_num(-1)  # Add 1 to avoid overlap with det_targets
    det_keypoints_xyc = dets.feats['keypoints_xyc'].cpu().flatten(-2, -1)
    track_keypoints_xyc = tracks.feats['keypoints_xyc'].cpu().flatten(-2, -1)

    confidence_threshold = 0.5
    if posetrack:
        joint_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                       (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]  # posetrack
    else:
        # h36m
        joint_pairs = [
            (0, 1),  # root to rhip
            (1, 2),  # rhip to rkne
            (2, 3),  # rkne to rank
            (0, 4),  # root to lhip
            (4, 5),  # lhip to lkne
            (5, 6),  # lkne to lank
            (0, 7),  # root to belly
            (7, 8),  # belly to neck
            (8, 9),  # neck to nose
            (9, 10),  # nose to head
            (8, 11),  # neck to lsho
            (11, 12),  # lsho to lelb
            (12, 13),  # lelb to lwri
            (8, 14),  # neck to rsho
            (14, 15),  # rsho to relb
            (15, 16)  # relb to rwri
        ]

    # Determine if the img_paths_tensor is a list of paths or a tensor of image data
    if image_paths is not None:
        num_batches = len(image_paths)
    else:
        num_batches = 1

    grid_size = int(np.ceil(np.sqrt(num_batches)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    if num_batches == 1:
        axes = np.array([axes])

    for i in range(grid_size * grid_size):
        if i >= num_batches:
            axes.flatten()[i].axis('off')
            continue

        ax = axes.flatten()[i]

        if img_tensor is not None:
            img_data = img_tensor[i].squeeze().cpu().numpy().astype(np.uint8)
            img = Image.fromarray(img_data).resize((1920, 1080))
        else:
            img_path = image_paths[i]
            img = Image.open(img_path).resize((1920, 1080))

        ax.imshow(img)
        ax.axis('off')

        all_targets = {target: i for i, target in enumerate(set(track_targets[i].int().tolist() + det_targets[i].int().tolist())) if target != -1}
        # k_to_t = {}
        # unique_targets = list(set(track_targets[i].numpy()).union(set(det_targets[i].numpy())))
        for j, (bbox_ltwh, masks, targets, keypoints) in enumerate(
                [(det_bbox_ltwh[i], det_masks[i], det_targets[i], det_keypoints_xyc[i]),
                 (track_bbox_ltwh[i], track_masks[i], track_targets[i], track_keypoints_xyc[i])]):
            style = '-' if j == 0 else '--'
            unique_targets = set(targets.numpy())
            if -1 in unique_targets:
                unique_targets.remove(-1)  # remove padding
            unique_targets = list(unique_targets)
            # print(f"unique_targets: {unique_targets}")
            for idx, t in enumerate(unique_targets):
                color = plt.cm.jet(all_targets[int(t)] / len(all_targets))
                indices = np.where(targets == t)
                assert len(indices[0]) == 1
                for k in indices[0]:
                    if masks[k]:
                        left, top, width, height = bbox_ltwh[k][0].cpu().numpy() * np.array([1920, 1080, 1920, 1080])
                        rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor=color,
                                                 facecolor='none', linestyle=style)
                        ax.add_patch(rect)
                        ax.text(left + width, top, str(k), color=color)
                        # k_to_t[f"{j}_{idx}_{k}_{t}"] = (left, top, width, height)

                        if j == 0 and num_batches != 1:
                            flag = 'V' if image_paths[i][k] == img_path else 'X'
                            ax.text(left, top, flag, color=color)

                        # Plot keypoints
                        kps = []
                        for kp in range(0, 51, 3):
                            # use the most recent skeleton in the tracklet:
                            kp_x = keypoints[k][0][kp] * 1920
                            kp_y = keypoints[k][0][kp + 1] * 1080
                            kp_c = keypoints[k][0][kp + 2]

                            if kp_c > confidence_threshold:
                                marker_style = 'o' if style == '-' else 'v'  # 'v' (triangle) for predictions
                                fill_style = color if style == '-' else 'none'  # If prediction, 'none' for empty triangle, else filled
                                ax.scatter(kp_x, kp_y, c=color, marker=marker_style, s=20, edgecolors=color,
                                           facecolors=fill_style)
                                kps.append((kp_x, kp_y))
                            else:
                                kps.append(None)

                        # Draw lines between keypoints to form the joints
                        for jp in joint_pairs:
                            if kps[jp[0]] and kps[jp[1]]:
                                joint_style = ':' if style == '--' else '-'
                                ax.plot([kps[jp[0]][0], kps[jp[1]][0]], [kps[jp[0]][1], kps[jp[1]][1]], color=color,
                                        linestyle=joint_style)
                    else:
                        print(f"skipping {k} because mask is False")

    plt.tight_layout()
    return plt
