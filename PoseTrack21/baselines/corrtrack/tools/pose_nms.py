import numpy as np
import os
import json
import argparse

from common.utils.pose_refine_utils import compute_OKS_PoseTrack_iou, compute_pose_area

def pose_nms(poses, 
             scores, 
             joint_threshold, 
             oks_threshold):
    poses[:, :, 2] = np.array(scores)
    pose_scores = np.mean(poses[:, :, 2], axis=1)
    order = pose_scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = compute_OKS_PoseTrack_iou(poses[i].reshape([-1]), 
                                            poses[order[1:]].reshape([-1, 51]), 
                                            joint_threshold)
        inds = np.where(oks_ovr < oks_threshold)[0]
        order = order[inds + 1]

    return keep

def perform_pose_nms(result_path, 
                     save_path, 
                     joint_threshold, 
                     oks_thres):

    save_path = save_path.format(joint_threshold, 
                                 oks_thres)

    os.makedirs(save_path, exist_ok=True)

    seq_files = os.listdir(result_path)
    for seq_file in seq_files:
        refined_annos = []
        with open(os.path.join(result_path, seq_file), 'r') as f:
            annos = json.load(f)

        seq_annos = {}
        for ann in annos['annotations']:
            im_id = ann['image_id']
            if im_id not in seq_annos:
                seq_annos[im_id] = []

            seq_annos[im_id].append(ann)

        for im_id, img_annos in seq_annos.items():
            kpts = [ann['keypoints'] for ann in img_annos]
            scores = [ann['scores'] for ann in img_annos]

            if len(kpts) > 1:
                kpts_ = np.array(kpts)
                kpts_ = kpts_.reshape([len(kpts), -1, 3])
                scores_ = np.array(scores)
                keep_poses = pose_nms(kpts_, scores_, joint_threshold, oks_thres)
            else:
                keep_poses = [0]

            refined_annos += [img_annos[idx] for idx in keep_poses]

        annos_before = len(annos['annotations'])
        annos_after = len(refined_annos)

        annos['annotations'] = refined_annos

        print("[{}]: Before: {} After: {}".format(seq_file, annos_before, annos_after))
        with open(os.path.join(save_path, seq_file), 'w') as f:
            json.dump(annos, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--joint_threshold', type=float, default=0.05)
    parser.add_argument('--oks_threshold', type=float, default=0.7)
    args = parser.parse_args()

    save_path = os.path.join(args.save_path, 'jt_{}_oks_{}_3_stage/sequences/')
    joint_threshold = args.joint_threshold
    oks_thres = args.oks_threshold

    perform_pose_nms(args.result_path, save_path, joint_threshold, oks_thres) 
