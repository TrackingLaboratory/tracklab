import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json
import random
import PIL

import os
import time
import cv2 as cv
import argparse


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from PIL import Image
from torchvision.transforms import functional as F

from models.correspondences.siamese_refinement_model import Siamese
from common.utils.pose_refine_utils import create_warp_matrix

from corrtrack.tracking.corr_tracking_w_reid_functions import *
from corrtrack.seqnet import get_seqnet_api, seqnet_api
from corrtrack.tracking import TrackManager, TrackList
from corrtrack.tracking.track_utils import get_reid_box_embeddings

class Sequence(Dataset):
    def __init__(self, anno, data_path, joint_threshold):

        self.anno = anno
        self.images_path = data_path
        self.output_size = [512, 512]
        self.detections_per_image = {}
        self.max_detections_per_image = 80
        self.ann_ids = []
        self.joint_threshold = joint_threshold
        for i, ann in enumerate(anno['annotations']):
            if not ann:
                continue
            if ann['image_id'] not in self.detections_per_image:
                self.detections_per_image[ann['image_id']] = []

            self.detections_per_image[ann['image_id']].append({'ann': ann, 'ann_id': i})
            self.ann_ids.append(0)

    def __len__(self):
        return len(self.anno['images'])

    def get_ann_ids(self):
        return self.ann_ids

    def bounding_box_from_pose(self, pose, height, width):
        valid = pose[2] >= self.joint_threshold

        if np.count_nonzero(valid) <= 1:
            return [0, 0, 0, 0]

        x_min = np.min(pose[0, valid])
        y_min = np.min(pose[1, valid])

        x_max = np.max(pose[0, valid])
        y_max = np.max(pose[1, valid])

        x_min = np.maximum(0, x_min - 10)
        y_min = np.maximum(0, y_min - 10)

        x_max = np.minimum(width - 1, x_max + 10)
        y_max = np.minimum(height - 1, y_max + 10)

        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, ind):
        im = self.anno['images'][ind]
        img = cv.imread(self.images_path + im['file_name'], 1)

        ###########################
        # Modification for SeqNet #
        ###########################
        pil_image = Image.open(self.images_path + im['file_name']).convert("RGB")
        pil_image_tensor = F.to_tensor(pil_image)
        ####### modification end ########

        height, width = img.shape[:2]
        crop_pos = [width // 2, height // 2]
        max_d = max(height, width)
        scales = [self.output_size[0] / float(max_d), 
                  self.output_size[1] / float(max_d)]

        t_form = create_warp_matrix(self.output_size, crop_pos, scales)
        im_cv = cv.warpAffine(img, 
                              t_form[0:2, :], 
                              (self.output_size[0], self.output_size[1]))

        img = cv.cvtColor(im_cv, cv.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()
        img = torch.transpose(img, 1, 2)
        img = torch.transpose(img, 0, 1)
        img /= 255

        joints_image = torch.ones(self.max_detections_per_image, 3, 17).mul(-1)
        bbs_image = torch.ones(self.max_detections_per_image, 4).mul(-1)
        original_joints = torch.ones(self.max_detections_per_image, 3, 17).mul(-1)

        ids = torch.zeros(self.max_detections_per_image)
        if im['id'] in self.detections_per_image:
            detections = self.detections_per_image[im['id']]
            for i, det in enumerate(detections):
                joints = np.ones((3, 17))
                joints[0, :] = det['ann']['keypoints'][0:51:3]
                joints[1, :] = det['ann']['keypoints'][1:51:3]
                scores = np.array(det['ann']['scores'])

                invalid_joints = scores < self.joint_threshold
                scores[invalid_joints] = 0
                joints[:, invalid_joints] = 0

                joints_warped = np.matmul(t_form, joints)
                joints_image[i, :2, :] = torch.from_numpy(joints_warped[:2, :])
                joints_image[i, 2, :] = torch.from_numpy(scores)

                joints[2] = scores
                bb = np.array(self.bounding_box_from_pose(joints, height, width))

                if bb[0] == 0 and bb[1] == 0 and bb[2] == 0 and bb[3] == 0:
                    joints_image[i, :2, :] = -1
                    joints_image[i, 2, :] = -1
                    continue

                bbs_image[i] = torch.from_numpy(bb)
                original_joints[i] = torch.from_numpy(joints)

                ids[i] = det['ann_id']

        return img, joints_image, ids, np.linalg.inv(t_form), bbs_image, original_joints, pil_image_tensor


def initialize_tracks(frame_data, features, min_keypoints, joint_threshold, seq_net, max_num_persons=80):
    track_manager = TrackManager.get_instance()

    keypoints_all = frame_data['kpts']
    ann_ids = frame_data['anno_ids']
    bboxes_from_pose = frame_data['bbs']

    box_embeddings, valid_boxes_inds = get_reid_box_embeddings(bboxes_from_pose,
                                                               seq_net)

    for k in range(max_num_persons):
        kpts = keypoints_all[k]

        # skip if pose is invalid
        if np.sum(kpts[2]) == -17:
            continue

        num_valid_kpts = np.count_nonzero(kpts[2] >= joint_threshold)

        # don't track if there are not enough points
        if num_valid_kpts < min_keypoints:
            continue

        if k not in valid_boxes_inds[:, 0]:
            continue

        embedding_idx = np.argwhere(valid_boxes_inds[:, 0] == k)[0]
        box_embedding = box_embeddings[embedding_idx]

        queries = build_querie(kpts)
        track = track_manager.new_track(queries=queries,
                                        kpts=kpts,
                                        ann_id=int(ann_ids[k].item()),
                                        features=features,
                                        curr_frame=0,
                                        num_kpts=num_valid_kpts,
                                        bbx=bboxes_from_pose[k],
                                        reid_features=box_embedding)
        track_manager.add(track)

    return track_manager.get_tracks()


def track_with_correspondences(
        ckpt_path, 
        sequences_path, 
        save_path, 
        dataset_path, 
        joint_threshold=.3,
        corr_threshold=.1,
        oks_threshold=.8,
        win_length=2,
        min_keypoints=2,
        min_track_len=4,
        duplicate_ratio=0.5,
        post_process_joint_threshold=0.1,
        # SeqNet related args
        seqnet_ckpt=None,
        inactive_patiences=None,
        similarity_threshold=None,
        min_refinement_track_len=1,
        min_consecutive_len=1):

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    ################################
    # SeqNet related configuration #
    ################################
    break_tracks = True     # we always break tracks
    assert seqnet_ckpt is not None and inactive_patiences is not None and similarity_threshold is not None
    TrackManager.set_inactive_patience(inactive_patiences)
    track_manager = TrackManager.get_instance()
    seq_net = get_seqnet_api(seqnet_ckpt)

    os.makedirs(save_path, exist_ok=True)
    seq_save_path = os.path.join(save_path,
                                 'jt_thres_{}_duplicate_ratio_{}_oks_{}_corr_threshold_{}_win_len_{}_min_keypoints_{}_min_track_len_{}_similarity_threshold_{}_pp_joint_threshold_{}_min_refinement_track_len_{}_min_consecutive_len_{}_inactive_patience_{}/sequences/'.format(
                                 joint_threshold,
                                 duplicate_ratio,
                                 oks_threshold,
                                 corr_threshold,
                                 win_length,
                                 min_keypoints,
                                 min_track_len,
                                 similarity_threshold,
                                 post_process_joint_threshold,
                                 min_refinement_track_len,
                                 min_consecutive_len,
                                 track_manager.inactive_patience))

    os.makedirs(seq_save_path, exist_ok=True)

    poseNet = Siamese(128)
    model = DataParallel(poseNet)
    model.cuda()
    checkpoint = torch.load(ckpt_path)
    pretrained_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for k_ckpt, v in pretrained_dict.items():
        new_k = k_ckpt
        if 'module.model.module' in k_ckpt:
            new_k = f"{k_ckpt[:13]}{k_ckpt[20:]}"

        model_state_dict[new_k] = v
    model.load_state_dict(model_state_dict)

    sequences = os.listdir(sequences_path)
    total_sequences = len(sequences)

    res_h = 128
    res_w = 128

    start_time = time.time()
    for seq_id, seq in enumerate(sequences):
        with open(sequences_path + seq, 'r') as f:
            anno = json.load(f)

            save_file_path = os.path.join(seq_save_path, seq)
            if os.path.isfile(save_file_path):
                print("Save file exists, skipping this one")
                continue

        print('Sequence : ' + str(seq_id) + '/' + str(total_sequences))
        print('Pre-processing')

        seq_data = Sequence(anno, dataset_path, joint_threshold)
        batch_size = win_length
        seq_data_loader = DataLoader(seq_data, batch_size=batch_size, shuffle=False, num_workers=4)

        images = []
        metadata = []
        features = []
        seqnet_image_tensors = []   # tensors for each images to be used by seqnet

        #########################################
        # Reset TrackManager for every sequence #
        #########################################
        track_manager = TrackManager.get_instance()
        track_manager.reset()

        for (images_batch,
             keypoints, 
             ids, 
             t_forms, 
             bbs_image, 
             original_kpts, 
             pil_images) in tqdm(seq_data_loader):
            with torch.no_grad():
                inputs = images_batch.cuda(non_blocking=True)
                f, _ = model(inputs, inputs, queries=None, embeddings_only=True)
                N = images_batch.shape[0]

                # add meta data
                for n in range(N):
                    images.append(images_batch[n])
                    seqnet_image_tensors.append(pil_images[n])

                    data = {'kpts': keypoints[n].numpy(),
                            'anno_ids': ids[n],
                            'warps': t_forms[n].numpy(),
                            'bbs': bbs_image[n].numpy(),
                            'original_kpts': original_kpts[n].numpy()}

                    metadata.append(data)
                    features.append(f[n])

        ann_ids = seq_data.get_ann_ids()

        with torch.no_grad():
            seq_net.load_image(seqnet_image_tensors[0].cuda())
            # initialize tracks
            tracks = initialize_tracks(metadata[0], 
                                       features[0], 
                                       min_keypoints, 
                                       joint_threshold, 
                                       seq_net,
                                       max_num_persons=seq_data.max_detections_per_image)

        num_frames = len(images)
        with torch.no_grad():
            for f in tqdm(range(1, num_frames)):
                frame_data = metadata[f]

                # load current frame into seqNet!
                seq_net.load_image(seqnet_image_tensors[f].cuda())
                # for each track, find correspondences
                affinities_clean, affinities, votes = get_affinities(tracks, 
                                                                     features, 
                                                                     model, 
                                                                     frame_data, 
                                                                     f,
                                                                     break_tracks, 
                                                                     joint_threshold, 
                                                                     corr_threshold,
                                                                     oks_threshold, 
                                                                     res_h, 
                                                                     res_w, 
                                                                     min_keypoints,
                                                                     seq_data.max_detections_per_image,
                                                                     images)

                tracks = perform_tracking(tracks, 
                                          affinities, 
                                          affinities_clean, 
                                          votes, 
                                          frame_data, 
                                          features,
                                          joint_threshold, 
                                          ann_ids, 
                                          min_keypoints, 
                                          f,
                                          seq_data.max_detections_per_image, 
                                          duplicate_ratio,
                                          break_tracks, 
                                          seq_net, 
                                          similarity_threshold, 
                                          min_refinement_track_len, 
                                          min_consecutive_len,
                                          images)

        track_counter = 0
        tracks = track_manager.get_tracks()
        annotations_w_track = []

        for track_id, track in enumerate(tracks):
            if len(track['ann_id']) < min_track_len:
                continue

            valid_annotations = []
            unique_frame_ids = []
            for ann_id in track['ann_id']:
                anno['annotations'][ann_id]['track_id'] = track_counter
                im_id = anno['annotations'][ann_id]['image_id']
                if im_id in unique_frame_ids:
                    assert False

                unique_frame_ids.append(im_id)
                keypoints = np.array(anno['annotations'][ann_id]['keypoints']).reshape([-1, 3])
                scores = np.array(anno['annotations'][ann_id]['scores'])

                invalid_joints = scores < post_process_joint_threshold

                # drop annotation
                if np.count_nonzero(invalid_joints) == keypoints.shape[0]:
                    continue

                scores[invalid_joints] = 0
                keypoints[invalid_joints, :] = 0

                anno['annotations'][ann_id]['keypoints'] = keypoints.reshape([-1]).tolist()
                anno['annotations'][ann_id]['scores'] = scores.tolist()

                valid_annotations.append(anno['annotations'][ann_id])

            if len(valid_annotations) < min_track_len:
                continue

            annotations_w_track += valid_annotations
            track_counter += 1

        anno['annotations'] = annotations_w_track

        with open(os.path.join(seq_save_path, seq), 'w') as outfile:
            json.dump(anno, outfile)

    print('total time: {}'.format(time.time() - start_time))

def run(args):

     track_with_correspondences(
         save_path=args.save_path,
         sequences_path=args.sequences_path,
         dataset_path=args.dataset_path,
         ckpt_path=args.ckpt_path,
         joint_threshold=args.joint_threshold,
         oks_threshold=args.oks_threshold,
         corr_threshold=args.corr_threshold,
         min_keypoints=args.min_keypoints,
         min_track_len=args.min_track_len,
         duplicate_ratio=args.duplicate_ratio,
         post_process_joint_threshold=args.post_process_joint_threshold,
         # SeqNet releated args
         seqnet_ckpt=args.seqnet_ckpt,
         inactive_patiences=args.inactive_patiences,
         similarity_threshold=args.similarity_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True) 
    parser.add_argument('--sequences_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)

    parser.add_argument('--joint_threshold', type=float, default=0.1)
    parser.add_argument('--oks_threshold', type=float, default=0.2)
    parser.add_argument('--corr_threshold', type=float, default=0.3)
    parser.add_argument('--min_keypoints', type=int, default=2)
    parser.add_argument('--min_track_len', type=int, default=3)
    parser.add_argument('--duplicate_ratio', type=float, default=0.6)
    parser.add_argument('--post_process_joint_threshold', type=float, default=0.3)

    #########################################
    # Parameters for SeqNet related options #
    #########################################
    parser.add_argument('--seqnet_ckpt', type=str, required=True) 
    parser.add_argument('--inactive_patiences', type=int, required=True)
    parser.add_argument('--similarity_threshold', type=float, required=True)
    parser.add_argument('--min_refinement_track_len', type=float, default=1)
    parser.add_argument('--min_consecutive_len', type=int, default=1)
    args = parser.parse_args()
    run(args)
