import torch
import random
import torch.backends.cudnn as cudnn
import argparse
import json
import numpy as np
import cv2
import os
import time

from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F

from models.pose_estimation.bn_inception2 import bninception
from common.utils.inference_utils import get_preds_for_pose, get_transform
from common.utils.data_utils import get_posetrack_eval_dummy

flipRef = [i - 1 for i in [1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]]

def apply_augmentation_test(example, img_dir, output_size=[256, 256]):
    im = cv2.imread(os.path.join(img_dir, example['image_location']), 1)

    x1, x2 = example['bbox'][0], example['bbox'][0] + example['bbox'][2]
    y1, y2 = example['bbox'][1], example['bbox'][1] + example['bbox'][3]

    crop_pos = [(x1 + x2) / 2, (y1 + y2) / 2]
    max_d = np.maximum(example['bbox'][2], example['bbox'][3])

    scales = [output_size[0] / float(max_d), output_size[1] / float(max_d)]

    param = {'rot': 0,
            'scale_x': 1,
            'scale_y': 1,
            'flip': 0,
            'tx': 0,
            'ty': 0}

    t_form = get_transform(param, crop_pos, output_size, scales)
    im_cv = cv2.warpAffine(im, t_form[0:2, :], (output_size[0], output_size[1]))
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    imf = cv2.flip(img, 1)

    img = torch.from_numpy(img).float()
    img = torch.transpose(img, 1, 2)
    img = torch.transpose(img, 0, 1)
    img /= 255

    imf = torch.from_numpy(imf).float()
    imf = torch.transpose(imf, 1, 2)
    imf = torch.transpose(imf, 0, 1)
    imf /= 255

    warp = torch.from_numpy(np.linalg.inv(t_form))

    return img, imf, warp


class PoseTrack:

    def __init__(self, 
                 args, 
                 img_dir=None, 
                 dtype='train', 
                 anno_file_path=None):

        self.output_size = [args.output_size_x, args.output_size_y]
        self.keep_track_id = args.keep_track_id
        
        with open(anno_file_path) as anno_file:
            self.anno = json.load(anno_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.anno)


    def __getitem__(self, ind):
        images, imagesf, warps = apply_augmentation_test(self.anno[ind], 
                                                         self.img_dir,
                                                         output_size=self.output_size)

        kpts = torch.from_numpy(np.array(self.anno[ind]['keypoints'])).float()
        bbx = torch.from_numpy(np.array(self.anno[ind]['bbox']))
        area = bbx[2] * bbx[3]

        meta = {'imgID': self.anno[ind]['img'], 
                'file_name': self.anno[ind]['image'],
                'warps': warps, 
                'bbox': bbx, 
                'seq_name': self.anno[ind]['seq_name'],
                'area': area}

        if self.keep_track_id:
            meta['track_id'] = self.anno[ind]['track_id']

        return {'images': images, 'imagesf': imagesf, 'meta': meta}


def estimate_poses(args, 
                   val_loader, 
                   pt_prefix, 
                   output_size, 
                   checkpoint_path):

    poseNet = bninception(num_stages=args.num_stages, 
                          out_ch=51, 
                          pretrained=False)

    poseNet = poseNet.cuda()
    checkpoint = torch.load(checkpoint_path)
    pretrained_dict = checkpoint['state_dict']
    dict_keys = list(pretrained_dict.keys())

    if 'module.' in dict_keys[0]:
        model = DataParallel(poseNet)
    else:
        model = poseNet
        
    model.load_state_dict(pretrained_dict)
    model.eval()

    sequences = {}
    seq_imgs = {}

    sequence_list = os.listdir(os.path.join(args.dataset_path, 
                                            'posetrack_data/', 
                                            pt_prefix))

    for seq_name in sequence_list:
        with open(os.path.join(args.dataset_path, 
                               'posetrack_data/', 
                               pt_prefix, 
                               seq_name), 'r') as f:
            anno = json.load(f)

        seq = seq_name.split('.')[0]
        seq_imgs[seq] = anno['images']

    seq_track_ctr = {}
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tqdm(val_loader)):

            images = sampled_batch['images']
            imagesf = sampled_batch['imagesf']
            inputs = images.cuda()
            inputsf = imagesf.cuda()

            output = model(inputs)
            output_det = torch.sigmoid(output[1][:, 0:17, :, :])
            output_det = output_det.data.cpu()

            outputf = model(inputsf)
            output_detf = torch.sigmoid(outputf[1][:, 0:17, :, :])
            output_detf = output_detf.data.cpu()

            sr = output[1][:, 17:51, :, :].data.cpu()
            N = output_det.shape[0]

            for n in range(N):
                size_y = args.output_size_y // 4
                size_x = args.output_size_x // 4

                prs = torch.zeros(17, size_y, size_x)
                output_detf[n] = output_detf[n][flipRef]
                for j in range(17):
                    prs[j] = output_det[n][j] + torch.from_numpy(cv2.flip(output_detf[n][j].numpy(), 1))

                keypoints, scores = get_preds_for_pose(prs, 
                                                       sampled_batch['meta']['warps'][n], sr[n],
                                                       output_size=output_size,
                                                       joint_scores=True)

                meta_data = sampled_batch['meta']
                sequence_name = meta_data['seq_name'][n]
                if sequence_name not in seq_track_ctr:
                    seq_track_ctr[sequence_name] = -1

                seq_track_ctr[sequence_name] += 1
                if sequence_name not in sequences:
                    sequences[sequence_name] = get_posetrack_eval_dummy()

                seq_data = sequences[meta_data['seq_name'][n]]

                im_id = meta_data['imgID'][n].item()

                track_id = seq_track_ctr[sequence_name] 
                if args.keep_track_id:
                    track_id = int(meta_data['track_id'][n])
                anno = {'image_id': im_id, 
                        'keypoints': keypoints, 
                        'scores': scores, 
                        'track_id': track_id,
                        'bbox': meta_data['bbox'][n][:-1].tolist()}

                seq_data['annotations'].append(anno)

    os.makedirs(args.save_path, exist_ok=True)

    for seq_name in seq_imgs.keys():
        if seq_name not in sequences.keys():
            sequences[seq_name] = get_posetrack_eval_dummy()

        images = seq_imgs[seq_name]
        sequence_anno = sequences[seq_name]

        sequence_anno['images'] = images

        with open(os.path.join(args.save_path, '{}.json'.format(seq_name)), 'w') as f:
            json.dump(sequences[seq_name], f)


def generate_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--annotation_file_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)

    parser.add_argument('--prefix', type=str, default='val')
    parser.add_argument('--joint_threshold', type=float, default=0.0)
    parser.add_argument('--output_size_x', type=int, default=288)
    parser.add_argument('--output_size_y', type=int, default=384)
    parser.add_argument('--num_stages', type=int, default=2)
    parser.add_argument('--keep_track_id', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    return args


def main():

    args = generate_parameters()
    cudnn.benchmark = True

    posetrack_val = PoseTrack(args=args,
                              img_dir=args.dataset_path,
                              dtype='val',
                              anno_file_path=args.annotation_file_path)

    val_loader = DataLoader(posetrack_val, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers)

    estimate_poses(args, 
                   val_loader, 
                   pt_prefix=args.prefix, 
                   output_size=[args.output_size_x, args.output_size_y],
                   checkpoint_path=args.checkpoint_path)

if __name__ == '__main__':
    main()
