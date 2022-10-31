import argparse
import json
import os

import torch
import numpy as np

from datasets import PoseTrack
from dekr2detections import DEKR2detections
from strong_sort2detections import StrongSORT2detections


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--posetrack', required=True, help='path to PoseTrack21/data/ folder')
    parser.add_argument('--project', default='val', help='project name')
    parser.add_argument('--name', default='exp', help='experience name')
    parser.add_argument('--train', action='store_true', help='evaluate on PoseTrack21 training set')
    parser.add_argument('--mot', action='store_true', help='evaluate MOT')
    parser.add_argument('--pose-estimation', action='store_true', help='evaluate pose estimation')
    parser.add_argument('--pose-tracking', action='store_true', help='evaluate pose tracking')
    parser.add_argument('--reid-pose-tracking', action='store_true', help='evaluate Re-ID pose tracking')
    parser.add_argument('--config-dekr', type=str, default='DEKR/experiments/inference.yaml')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/track.yaml')
    args = parser.parse_args()
    return args


@torch.no_grad()
def val(
    posetrack,
    project='val',
    name='exp',
    train=False,
    mot=False,
    pose_estimation=False,
    pose_tracking=False,
    reid_pose_tracking=False,
    config_dekr='DEKR/experiments/inference.yaml',
    config_strongsort='strong_sort/configs/track.yaml',
):
    # handle args
    assert any((mot, pose_estimation, pose_tracking, reid_pose_tracking)), \
        "At least one evaluation method should be selected with --mot, " +\
        "--pose-estimation, --pose-tracking or --reid-pose-tracking"
    track_required = any((mot, pose_tracking, reid_pose_tracking))
    
    # handle paths
    save_path = os.path.join('runs', project, name)
    i = 0
    while os.path.exists(save_path + str(i)):
        i += 1
    save_path = save_path + str(i)
    os.makedirs(save_path, exist_ok=True)
    if mot:
        mot_folder = os.path.join(save_path, "mot")
        os.makedirs(mot_folder, exist_ok=True)
    if pose_estimation:
        pose_est_folder = os.path.join(save_path, "pose_estimation")
        os.makedirs(pose_est_folder, exist_ok=True)
    if any((pose_tracking, reid_pose_tracking)):
        pose_track_folder = os.path.join(save_path, "pose_tracking")
        os.makedirs(pose_track_folder, exist_ok=True)
    
    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load pose extractor 
    # For once, same for every videos
    # TODO use this as a module, not always DEKR but whatever we want
    model_pose = DEKR2detections(
        config_dekr, 
        device,
        vis_threshold=0.3 # TODO maybe add to .yaml file ?
    )
    
    # TODO replace by @Vlad framwork and make it modulable
    # TODO use this as a module, not always StrongSORT but whatever we want
    model_track = StrongSORT2detections(
        config_strongsort,
        device
    )
        
    # load dataloader
    dataset = PoseTrack(posetrack)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    # process images
    mot_str = ""
    list_images = []
    list_est_anns = []
    list_track_anns = []
    dumb_id = 0
    for i, data in enumerate(dataloader): # image is Tensor RGB (1, 3, H, W)
        print(f"Frame {i+1}/{len(dataloader)}")
        
        # pose estimation part -> create detections object
        detections = model_pose.run(data['image'])
        
        # tracking part
        # 1. update detections object
        # 2. update results for posetrack evaluation
        if track_required: # update results w/ tracking
            detections = model_track.run(data['image'], detections)
            for Track in detections.Tracks:
                if mot:
                    mot_str += f"{data['frame'].item()}, {Track['ID']}, {Track['Bbox'][0]}, "+\
                        f"{Track['Bbox'][1]}, {Track['Bbox'][2]-Track['Bbox'][0]}, "+\
                        f"{Track['Bbox'][3]-Track['Bbox'][1]}, {Track['score']}, -1, -1, -1\n"
                if any((pose_tracking, reid_pose_tracking)):
                    Keypoints = np.asarray(Track['Keypoints']).flatten()
                    list_track_anns.append({
                        "bbox": [float(Track['Bbox'][0]),
                                 float(Track['Bbox'][1]),
                                 float(Track['Bbox'][2] - Track['Bbox'][0]),
                                 float(Track['Bbox'][3] - Track['Bbox'][1])],
                        "image_id": data['image_id'].item(),
                        "keypoints": Keypoints.astype(float).tolist(),
                        "scores": Keypoints[2::3].astype(float).tolist(),
                        "person_id": int(Track['ID']),
                        "track_id": int(Track['ID']),
                    })
        # update results w/o tracking
        if pose_estimation:
            for Bbox, Pose in zip(detections.Bboxes, detections.Poses):
                Keypoints = np.asarray(Pose).flatten().astype(float)
                list_est_anns.append({
                    "bbox": [float(Bbox[0]), float(Bbox[1]), 
                             float(Bbox[2] - Bbox[0]),
                             float(Bbox[3] - Bbox[1])],
                    "image_id": data['image_id'].item(),
                    "keypoints": Keypoints.astype(float).tolist(),
                    "scores": Keypoints[2::3].astype(float).tolist(),
                    "person_id": dumb_id,
                    "track_id": dumb_id,
                })
                dumb_id += 1
        if any((pose_estimation, pose_tracking, reid_pose_tracking)):
            list_images.append({
                "file_name": data['file_name'][0],
                "id": data['image_id'].item(),
                "image_id": data['image_id'].item(),
            })

        # if last frame -> write results to files (& load new model tracker)
        if data['frame'] == data['nframes']:
            ## TODO change this because it is ugly and not optimized
            #if track_required: 
            #    model_track = StrongSORT2detections(
            #        config_strongsort,
            #        device
            #    )
            if mot:
                mot_file_path = os.path.join(mot_folder, data['folder'][0]+'.txt')
                file = open(mot_file_path, "w+")
                file.write(mot_str)
                file.close()
                mot_str = ""
            if pose_estimation:
                output = {
                    "images": list_images,
                    "annotations": list_est_anns,
                }
                pose_path_file = os.path.join(pose_est_folder, data['folder'][0]+'.json')
                file = open(pose_path_file, "w+")
                json.dump(output, file)
                file.close()
                list_est_anns = []
            if any((pose_tracking, reid_pose_tracking)):
                output = {
                    "images": list_images,
                    "annotations": list_track_anns,
                }
                pose_path_file = os.path.join(pose_track_folder, data['folder'][0]+'.json')
                file = open(pose_path_file, "w+")
                json.dump(output, file)
                file.close()
                list_track_anns = []
            list_images = []
            
    # TODO add automatic lauch of posetrack evaluation on the files


def main():
    args = parse_args()
    val(**vars(args))

   
if __name__ == "__main__":
    main()