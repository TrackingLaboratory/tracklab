import argparse
from pathlib import Path
import os
import cv2

import numpy as np
import torch
import torch.utils.data

from datasets import ImageFolder
from dekr2detections import DEKR2detections

#import os
#import sys
#FILE = Path(__file__).resolve()
#ROOT = FILE.parents[0]  # yolov5 strongsort root directory
#WEIGHTS = ROOT / 'weights'
#
#if str(ROOT) not in sys.path:
#    sys.path.append(str(ROOT))  # add ROOT to PATH
#if str(ROOT / 'yolov5') not in sys.path:
#    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
#if str(ROOT / 'strong_sort') not in sys.path:
#    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from strong_sort.utils.parser import get_config # TODOs in here
from strong_sort.strong_sort import StrongSORT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', required=True, help='path to folder containing images to process')
    parser.add_argument('--project', default='track', help='project name')
    parser.add_argument('--name', default='exp', help='experience name')
    parser.add_argument('--show-poses', default=True, help='show keypoints')
    parser.add_argument('--show-tracks', default=True, help='show tracking results')
    parser.add_argument('--save-imgs', default=True, help='save images')
    parser.add_argument('--save-vid', default=True, help='save video')
    parser.add_argument('--config-dekr', type=str, default='DEKR/experiments/inference.yaml')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/track.yaml')
    args = parser.parse_args()
    return args

@torch.no_grad()
def track(
    input_folder,
    project='track',
    name='exp',
    show_poses=True,
    show_tracks=True,
    save_imgs=True,
    save_vid=True,
    config_dekr='DEKR/experiments/inference.yaml',
    config_strongsort='strong_sort/configs/track.yaml',
):
    # handle paths
    if save_imgs or save_vid:
        save_path = os.path.join('runs', project, name)
        i = 0
        while os.path.exists(save_path + str(i)):
            i += 1
        save_path = save_path + str(i)
        os.makedirs(save_path, exist_ok=True)
        if save_imgs:
            imgs_name = os.path.join(save_path, 'imgs')
            os.makedirs(imgs_name, exist_ok=True)
        if save_vid:
            vid_name = None
    
    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load pose extractor # TODO use this as a module, not always DEKR but whatever we want
    model_pose = DEKR2detections(
        config_dekr, 
        device,
        vis_threshold=0.3 # TODO maybe add to .yaml file ?
    )
    
    # load strongsort
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)
    weights = Path(cfg.STRONGSORT.WEIGHTS).resolve()
    
    model_tracking = StrongSORT(
        weights,
        device,
        cfg.STRONGSORT.HALF,
        max_dist=cfg.STRONGSORT.MAX_DIST,
        max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
        max_age=cfg.STRONGSORT.MAX_AGE,
        n_init=cfg.STRONGSORT.N_INIT,
        nn_budget=cfg.STRONGSORT.NN_BUDGET,
        mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
        ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
    ) # TODO use this as a module, not always StrongSORT but whatever we want
    
    # load dataloader
    dataset = ImageFolder(input_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    curr_frame, prev_frame = None, None
    # process images
    for i, image in enumerate(dataloader): # image is Tensor RGB (1, 3, H, W)
        
        # pose estimation part -> create detections object
        input2detect = model_pose.image2input(image) # -> (h, w, 3)
        detections = model_pose.estimate(input2detect)
        
        # tracking part -> update detections object
        # TODO replace by @Vlad framwork and make it modulable
        
        # preprocess input
        input2tracking = image[0].cpu().numpy() # -> (3, H, W)
        input2tracking = np.transpose(input2tracking, (1, 2, 0)) # -> (H, W, 3)
        input2tracking = input2tracking*255.0
        input2tracking = input2tracking.astype(np.uint8) # -> to uint8
        
        # camera compensation stuff
        # TODO not sure about utility
        curr_frame = input2tracking
        if cfg.STRONGSORT.ECC:  # camera motion compensation
            model_tracking.tracker.camera_update(prev_frame, curr_frame)
        
        # do tracking
        H, W = input2tracking.shape[:2]
        detections.update_HW(H, W)
        outputs = []
        if detections.scores: # check if instance(s) is detected
            Bboxes, confidences, classes = detections.get_StrongSORT_inputs()
            outputs = model_tracking.update(Bboxes,
                                            confidences,
                                            classes,
                                            input2tracking)
            
        print(f"Frame {i}/{len(dataloader)-1}:")
        print(f"Pose extractor detected {len(detections.scores)} person(s)")
        print(f"Tracking detected {len(outputs)} person(s)\n")
        
        if save_imgs or save_vid:
            detections.show_image(image)
            
            if show_poses:
                detections.show_Poses()
                detections.show_Bboxes()
            if show_tracks:
                detections.show_Tracks(outputs)
            
            img = detections.get_image()
            if save_imgs:
                path = os.path.join(imgs_name, f"{i}.jpg")
                cv2.imwrite(path, img)
            
            if save_vid:
                if not vid_name:
                    vid_name = os.path.join(save_path, 'results.mp4')
                    video = cv2.VideoWriter(vid_name, 
                                            cv2.VideoWriter_fourcc(*'mp4v'), 
                                            10,
                                            (W, H))
                video.write(img)
        prev_frame = curr_frame
    
    if save_vid:
        video.release()
    
def main():
    args = parse_args()
    track(**vars(args))
    
    
if __name__ == "__main__":
    main()