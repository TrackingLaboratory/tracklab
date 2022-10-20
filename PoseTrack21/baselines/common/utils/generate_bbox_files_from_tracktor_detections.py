import os
import json
import numpy as np
import argparse 

from pycocotools.coco import COCO
from tqdm import tqdm

def build_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--anno_path', type=str, default='posetrack_data/val')
    parser.add_argument('--bbox_file', type=str, required=True) 
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--bbox_thresh', type=float, default=0.5)
    parser.add_argument('--keep_track_id', type=bool, default=False, help='For MOT approaches, we want to keep the track id')

    args = parser.parse_args()

    return args 

def const():
    bb_files = '/media/work2/doering/2020/code/tracking_wo_bnw/output/faster_rcnn_fpn/new_dataset/faster_rcnn_fpn_training_mot_posetrack_PRETRAINED_test/det/detections.json'

def main():

    args = build_args()
    assert os.path.exists(args.dataset_path)
    assert os.path.exists(args.bbox_file)

    with open(args.bbox_file, 'r') as f:
        sequence_bbs = json.load(f)

    rec = []

    annotation_path = os.path.join(args.dataset_path, args.anno_path)
    sequence_files = os.listdir(annotation_path)
    for seq_idx, file in enumerate(tqdm(sequence_files)):
        seq_name = os.path.splitext(file)[0]

        api = COCO(os.path.join(annotation_path, file))
        img_ids = api.getImgIds()
        imgs = api.loadImgs(img_ids)

        for img_idx, img in enumerate(imgs):
            estimated_bbs = np.array(sequence_bbs[seq_name][str(img_idx+1)])[0]

            if len(estimated_bbs) > 0:
                estimated_bb_scores = np.asarray(estimated_bbs[:, -1])
                indexes = estimated_bb_scores > args.bbox_thresh
                estimated_bbs = estimated_bbs[indexes]
                estimated_bb_scores = estimated_bb_scores[indexes]

            for bb_idx, bb in enumerate(estimated_bbs):
                # faster rcnn provides boxes with x1, y1, x2, y2
                # posetrack gt boxes provided as x1, y1, w, h

                track_id = -1
                if args.keep_track_id:
                    # bb contains track id at first position! 
                    track_id = int(bb[0])
                    bb = bb[1:]

                bb[2] -= bb[0]
                bb[3] -= bb[1]
                sc = np.maximum(bb[2], bb[3])

                if sc == 0:
                    continue

                if bb[0] < 0:
                    bb[0] = 0

                if bb[1] < 0:
                    bb[1] = 0

                if bb[2] < 0 or bb[3] < 0:
                    continue

                rec.append({
                    'image_location': img['file_name'],
                    'image': os.path.join(posetrack_home_fp, 'posetrack_data', img['file_name']),
                    'keypoints': [],
                    'bbox': bb.tolist(),
                    'bbox_score': estimated_bb_scores[bb_idx],
                    'img': img['id'],
                    'track_id': track_id,
                    'file_id': file,
                    'vid_id': img['vid_id'],
                    'seq_name': img['file_name'].split('/')[-2],
                    'frame_idx': img_idx,
                    'tot_frames': len(imgs)
                })

    os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/PoseTrack21_tracktor_bb_thres_{args.bbox_thresh}.json", 'w') as write_file:
        json.dump(rec, write_file)
