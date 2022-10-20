import json
import random
import os
import torch
import matplotlib.pyplot as plt
import time
import argparse
import cv2 as cv

import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from tqdm import tqdm

from common.utils.pose_refine_utils import *
from models.correspondences.siamese_refinement_model import Siamese
from models.pose_estimation.bn_inception2 import bninception

flipRef = [i - 1 for i in [1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]]


class Sequence:

    def __init__(self, anno, data_path):

        self.anno = anno
        self.images_path = data_path
        self.output_size = [512, 512]
        self.detections_per_image = {}
        self.max_detections_per_image = 100
        self.tmp_images = []

        for ann in anno['annotations']:
            if not ann:
                continue
            if ann['image_id'] not in self.detections_per_image:
                self.detections_per_image[ann['image_id']] = []

            self.detections_per_image[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.anno['images'])

    def __getitem__(self, ind):
        im = self.anno['images'][ind]
        img_orig = cv.imread(self.images_path + im['file_name'], 1)

        height, width = img_orig.shape[:2]
        crop_pos = [width // 2, height // 2]
        max_d = max(height, width)
        scales = [self.output_size[0] / float(max_d), 
                  self.output_size[0] / float(max_d)]

        t_form = create_warp_matrix(self.output_size, 
                                    crop_pos, 
                                    scales)

        im_cv = cv.warpAffine(img_orig, 
                              t_form[0:2, :], 
                              (self.output_size[0], self.output_size[1]))
        img = cv.cvtColor(im_cv, cv.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()
        img = torch.transpose(img, 1, 2)
        img = torch.transpose(img, 0, 1)
        img /= 255

        joints_image = torch.ones(self.max_detections_per_image, 3, 17).mul(-1)
        joints_orig = torch.ones(self.max_detections_per_image, 2, 17).mul(-1)
        track_ids = torch.ones(self.max_detections_per_image).mul(-1)

        total_instances = 0
        if im['id'] in self.detections_per_image:
            detections = self.detections_per_image[im['id']]
            total_instances = len(detections)
            for i, det in enumerate(detections):

                joints = np.ones((3, 17))
                joints[0, :] = det['keypoints'][0:51:3]
                joints[1, :] = det['keypoints'][1:51:3]
                joints_orig[i] = torch.from_numpy(joints[:2, :])
                joints_warped = np.matmul(t_form, joints)
                joints_image[i, :2, :] = torch.from_numpy(joints_warped[:2, :])
                joints_image[i, 2, :] = torch.from_numpy(np.array(det['scores']))
                track_ids[i] = det['track_id']

        return (img, 
                t_form, 
                joints_image, 
                im['id'], 
                total_instances, 
                img_orig, 
                joints_orig, 
                track_ids)


def correspondence_boxes(corr_ckpt_path,
                         pose_ckpt_path,
                         sequences_path,
                         save_path,
                         dataset_path,
                         bb_thres,
                         joint_threshold,
                         corr_threshold,
                         oks_threshold,
                         network_res,
                         pose_output_size,
                         min_kpts=3):

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    seq_save_path = os.path.join(
        save_path, 'val_set_bb_{}_jt_{}_with_corr_{}_at_oks_{}/sequences/'.format(bb_thres,
                                                                                  joint_threshold,
                                                                                  corr_threshold,
                                                                                  oks_threshold))

    os.makedirs(seq_save_path, exist_ok=True)

    poseNet = Siamese(res=128)
    model = DataParallel(poseNet)
    checkpoint = torch.load(corr_ckpt_path)
    pretrained_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for k_ckpt, v in pretrained_dict.items():
        new_k = k_ckpt
        if 'module.model.module' in k_ckpt:
            new_k = f"{k_ckpt[:13]}{k_ckpt[20:]}"

        model_state_dict[new_k] = v
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    poseNet2 = bninception(num_stages=args.num_stages, pretrained=False)
    poseNet2.cuda()
    checkpoint = torch.load(pose_ckpt_path)

    pretrained_dict = checkpoint['state_dict']
    dict_keys = list(pretrained_dict.keys())
    if 'module.' in dict_keys[0]:
        model2 = DataParallel(poseNet2)
    else:
        model2 = poseNet2
    model2.load_state_dict(pretrained_dict)
    model2.eval()

    sequences = os.listdir(sequences_path)
    total_sequences = len(sequences)
    res_h = network_res[1]
    res_w = network_res[0]

    start_time = time.time()
    total_added = 0

    for seq_id, seq in enumerate(sequences):

        with open(sequences_path + seq, 'r') as f:
            anno = json.load(f)

        print('Sequence : ' + str(seq_id) + '/' + str(total_sequences))
        print('Pre-processing')

        Seq = Sequence(anno, dataset_path)
        batch_size = 2
        Seq_loader = DataLoader(Seq, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=8)
        features = []
        metadata = []
        images = []
        images_orig = []

        for _, (images_batch, 
                warps, 
                keypoints, 
                im_ids, 
                N_ins, 
                img_orig, 
                joints_orig, 
                track_ids) in enumerate(tqdm(Seq_loader)):

            with torch.no_grad():
                inputs = images_batch.cuda(non_blocking=True)
                f, _ = model(inputs, 
                             inputs, 
                             queries=None,  
                             embeddings_only=True)

                N = images_batch.shape[0]
                for n in range(N):
                    metadata.append([
                        keypoints[n].numpy(), 
                        im_ids[n].item(), 
                        warps[n],
                        N_ins[n].item(), 
                        joints_orig[n], 
                        track_ids[n]
                    ])
                    features.append(f[n])
                    images.append(images_batch[n])
                    images_orig.append(img_orig[n])

        print('Computing correlations')
        N = len(features)
        new_annos_seq = []

        for n in tqdm(range(1, N)):
            prevdata = metadata[n-1]
            keypoints_all = prevdata[0]
            boxes_im = []
            centers_im = []
            N_ins_im = prevdata[3]
            track_ids_prev = prevdata[5]
            track_ids_im = []

            for k in range(N_ins_im):
                keypoints = keypoints_all[k]
                if int(keypoints[2, :].sum()) == -17:
                    continue

                queries = np.zeros((1, 17, 3))
                queries[0, :, 0] = (keypoints[1, :] / 4).astype(np.uint)
                queries[0, :, 1] = (keypoints[0, :] / 4).astype(np.uint)
                queries[0, :, 2] = keypoints[2, :]

                fA = features[n-1]
                fA = fA.view(1, 32, res_h, res_w)

                with torch.no_grad():
                    output = model(fA, 
                                   features[n].view(1, 32, res_h, res_w), 
                                   queries, 
                                   False, 
                                   True)

                    correlations = torch.sigmoid(output)
                    correlations2 = correlations.view(1, 17, res_h * res_w).data.cpu()
                    val_k, ind = correlations2.topk(1, dim=2)
                    xs = ind % res_w
                    ys = (ind / res_w).long()
                    val_k, xs, ys = val_k.view(1, 17).numpy(), xs.view(1, 17).numpy(), ys.view(1, 17).numpy()
                    poses, bbs, centres = get_pose_from_correspondences(ys, 
                                                                        xs, 
                                                                        val_k, 
                                                                        queries[0], 
                                                                        joint_threshold,
                                                                        corr_threshold,
                                                                        [network_res[0] * 4, network_res[1] * 4],
                                                                        min_kpts=min_kpts)

                    if bbs[0][0] == 0 and bbs[0][1] == 0:
                        continue
                    w, h = bbs[0][2] - bbs[0][0], bbs[0][3] - bbs[0][1]
                    if w == 0 or h == 0:
                        continue
                    boxes_im.append(bbs[0])
                    centers_im.append(centres[0])
                    track_ids_im.append(track_ids_prev[k])

            if len(boxes_im) > 0:

                warps_im = create_warps(boxes_im, pose_output_size)
                N_boxes = len(boxes_im)
                ims = torch.zeros(N_boxes, 3, pose_output_size[1], pose_output_size[0])
                imsf = torch.zeros(N_boxes, 3, pose_output_size[1], pose_output_size[0])
                warps_inv = []
                inp_images = []

                for i, w in enumerate(warps_im):
                    inp_img = images[n].permute(1, 2, 0).numpy()

                    im_cv = cv.warpAffine(inp_img, w[0:2, :],
                                          (pose_output_size[0], pose_output_size[1]))

                    img = torch.from_numpy(im_cv).float()
                    img = img.permute(2, 0, 1)
                    ims[i] = img

                    imf = cv.flip(im_cv, 1)
                    imgf = torch.from_numpy(imf).float()
                    imgf = imgf.permute(2, 0, 1)
                    imsf[i] = imgf

                    inp_images.append(inp_img)
                    warps_inv.append(np.linalg.inv(w))

                with torch.no_grad():
                    inputs = ims.cuda(non_blocking=True)
                    output = model2(inputs)
                    output_det = torch.sigmoid(output[1][:, 0:17, :, :])
                    output_det = output_det.data.cpu()
                    sr = output[1][:, 17:51, :, :].data.cpu()

                    inputsf = imsf.cuda(non_blocking=True)
                    outputf = model2(inputsf)
                    outputf = torch.sigmoid(outputf[1][:, 0:17])
                    outputf = outputf.data.cpu()

                for i, w_inv in enumerate(warps_inv):
                    prs = torch.zeros(17, pose_output_size[1] // 4, pose_output_size[0] // 4)
                    outputflip = outputf[i]
                    outputflip = outputflip[flipRef]

                    for j in range(17):
                        prs[j] = output_det[i][j] + torch.from_numpy(cv.flip(outputflip[j].numpy(), 1))

                    k, score, pose_crop, pwarped = get_preds(prs, w_inv, sr[i],
                                                             res_h=pose_output_size[1] // 4,
                                                             res_w=pose_output_size[0] // 4)

                    if np.count_nonzero(np.array(score) >= joint_threshold) < 5:
                        continue

                    pose_im, pose_orig_im, gt = get_poses(k, 
                                                          score, 
                                                          metadata[n][2].numpy(),
                                                          joint_threshold, 
                                                          [network_res[0] * 4, network_res[1] * 4])

                    max_oks = get_max_oks(metadata[n][4], N_ins_im, gt, joint_threshold)

                    if max_oks <= oks_threshold:
                        new_anno = create_new_anno(metadata[n][1], 
                                                   gt, 
                                                   score, 
                                                   track_ids_im[i])

                        metadata[n][0][N_ins_im] = torch.from_numpy(pose_im)
                        metadata[n][4][N_ins_im] = torch.from_numpy(pose_orig_im[:2, :])

                        N_ins_im += 1
                        new_annos_seq.append(new_anno)

        print('Total annos before :' + str(len(anno['annotations'])))
        for new_anno in new_annos_seq:
            anno['annotations'].append(new_anno)
            total_added += 1

        track_id_ctr = 0
        for ann in anno['annotations']:
            ann['track_id'] = track_id_ctr
            track_id_ctr += 1

        print('Total annos after :' + str(len(anno['annotations'])))
        with open(os.path.join(seq_save_path, seq), 'w') as outfile:
            json.dump(anno, outfile)

    print('Total new annos : ' + str(total_added))
    print('Total time : ' + str(time.time() - start_time))


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-oks', '--oks_threshold', type=float, default=0.7)
    parser.add_argument('-corr', '--corr_threshold', type=float, default=0.1)
    parser.add_argument('-jt', '--joint_threshold', type=float, default=0.05)
    parser.add_argument('--bb_thres', type=float, default=0.5)
    parser.add_argument('--corr_ckpt_path', type=str)
    parser.add_argument('--pose_ckpt_path', type=str)

    parser.add_argument('--sequences_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--num_stages', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    correspondence_boxes(corr_ckpt_path=args.corr_ckpt_path,
                         pose_ckpt_path=args.pose_ckpt_path,
                         sequences_path=args.sequences_path,
                         save_path=args.save_path,
                         dataset_path=args.dataset_path,
                         bb_thres=args.bb_thres,
                         joint_threshold=args.joint_threshold,
                         corr_threshold=args.corr_threshold,
                         oks_threshold=args.oks_threshold,
                         network_res=[128, 128],
                         pose_output_size=[288, 384])
