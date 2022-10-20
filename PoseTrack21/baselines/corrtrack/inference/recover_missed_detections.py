import json
import os
import time
import numpy as np
import cv2 as cv
import argparse

from common.utils.pose_refine_utils import compute_OKS_PoseTrack_tracking, get_max_oks
from tqdm import tqdm

def bounding_box_from_pose(x, y, s, th):
    valid = s > th

    x_min = np.min(x[valid])
    y_min = np.min(y[valid])

    x_max = np.max(x[valid])
    y_max = np.max(y[valid])
    w = x_max - x_min
    h = y_max - y_min
    a = w * h

    return a

def get_images(anno):
    images = {}

    for im in anno['images']:
        if im['id'] not in images:
            images[im['id']] = []

    return images


def get_annos_per_image(anno, images):
    for ann in anno['annotations']:
        if ann['image_id'] in images:
            images[ann['image_id']].append(ann)

    return images


def get_new_annos(dets, joint_thres, oks_thres):
    new_annos = []
    new_anns_im = []
    old_anns_im = []

    for det in dets:
        if 'new_anno' in det:
            new_anns_im.append(det)
        else:
            old_anns_im.append(det)

    if len(old_anns_im) == 0:
        for ann in new_anns_im:
            new_annos.append(ann)
    else:
        for new_ann in new_anns_im:

            X = new_ann['keypoints'][0::3]
            Y = new_ann['keypoints'][1::3]
            s = new_ann['scores']

            gt = []
            for j in range(17):
                gt.append(X[j])
                gt.append(Y[j])
                gt.append(s[j])

            max_OKS = 0
            dts = []

            for ann2 in old_anns_im:

                X = ann2['keypoints'][0::3]
                Y = ann2['keypoints'][1::3]
                s = ann2['scores']
                dt = []
                num_valid = 0
                for j in range(17):
                    dt.append(X[j])
                    dt.append(Y[j])
                    dt.append(s[j])
                    if s[j] > joint_thres:
                        num_valid += 1
                dts.append(dt)

                if num_valid > 2:
                    OKS = compute_OKS_PoseTrack_tracking(np.array(gt), np.array(dt), joint_thres)
                else:
                    OKS = 0

                if OKS > max_OKS:
                    max_OKS = OKS

            if max_OKS <= oks_thres:
                new_annos.append(new_ann)

    return new_annos


def recover_missed_detections(sequences_path, dataset_path, save_path, OKS_th, joint_th):
    sequences = os.listdir(sequences_path)
    seq_save_path = os.path.join(save_path, 
                                 'recover_missed_detections_jt_th_{}_oks_{}/sequences/'.format(joint_th, OKS_th))
    os.makedirs(seq_save_path, exist_ok=True)
    start_time = time.time()
    counter = 0

    for seq_id, seq in enumerate(sequences):
        print('Processing Seq : ' + str(seq_id))

        with open(sequences_path + seq, 'r') as f:
            anno = json.load(f)

        new_annos_seq = []
        for ann in anno['annotations']:
            if 'new_anno' in ann:
                continue
            new_annos_seq.append(ann)

        print('Old annos : ' + str(len(new_annos_seq)))

        image_infos = {img['id']: img['file_name'] for img in anno['images']}

        images = get_images(anno)
        images = get_annos_per_image(anno, images)

        new_annos_c = 0
        for image_id, dets in images.items():
            new_annos = get_new_annos(dets, joint_th, OKS_th)
            new_annos_c += len(new_annos)
            for ann in new_annos:
                new_annos_seq.append(ann)

        print("New annos {}".format(new_annos_c))

        anno['annotations'] = new_annos_seq
        with open(seq_save_path + seq, 'w') as outfile:
            json.dump(anno, outfile)

    print('total time : ' + str(time.time() - start_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences_path', type=str,   required=True)
    parser.add_argument('--dataset_path',   type=str) 
    parser.add_argument('--save_path',      type=str,   required=True)
    parser.add_argument('--oks',            type=float, default=0.6)
    parser.add_argument('--warp_oks',       type=float, default=0.8)
    parser.add_argument('--joint_th',       type=float, default=0.1)

    args = parser.parse_args()

    recover_missed_detections(args.sequences_path, 
                              args.dataset_path, 
                              args.save_path, 
                              args.oks, 
                              args.joint_th)
