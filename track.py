import os
import yaml
import argparse
from tqdm import tqdm

import torch

from tracker import Tracker
from datasets import ImageFolder
from vis_engine import VisEngine
from dekr2detections import DEKR2detections
from strong_sort2detections import StrongSORT2detections


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str,
                        help='path to directory containing images')
    parser.add_argument('--vis-cfg', type=str,
                        default='vis.yaml',
                        help='path to visualization config file')
    parser.add_argument('--dekr-cfg', type=str, 
                        default='dekr.yaml', 
                        help='path to dekr config file')
    parser.add_argument('--strongsort-cfg', type=str, 
                        default='strong_sort/configs/track.yaml', 
                        help='path to strongsort config file')
    args = parser.parse_args()
    return args


@torch.no_grad()
def track(
    input_dir,
    vis_cfg,
    dekr_cfg,
    strongsort_cfg,
    *args, **kwargs
):
    # handle vis_cfg and paths
    with open(vis_cfg, 'r') as f:
        vis_cfg = yaml.load(f, Loader=yaml.FullLoader)
    save_dir = os.path.join('runs', vis_cfg['save_dir'])
    if os.path.exists(save_dir):
        i = 0
        while os.path.exists(save_dir + str(i)): i += 1
        save_dir = save_dir + str(i)
        print(f"Save directory ({os.path.join('runs', vis_cfg['save_dir'])}) already exists.")
        print(f"New save directory replaced by ({save_dir}).")
    os.makedirs(save_dir, exist_ok=True)
    
    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load pose extractor 
    # TODO make it modulable and more parametrable
    model_pose = DEKR2detections(
        dekr_cfg, 
        device,
        vis_threshold=0.3 # FIXME add vis channel
    )
    
    # load model tracker
    # TODO make it modulable and more parametrable
    model_track = StrongSORT2detections(
        strongsort_cfg,
        device
    )
    
    # load dataset
    dataset = ImageFolder(input_dir)
    
    # process images
    all_detections = []
    for data in tqdm(dataset, desc='Inference'): # tensor RGB (3, H, W)
        # pose estimation part -> create detections object
        detections = model_pose.run(data)
        
        # tracking part -> update detections object
        detections = model_track.run(data, detections)
        all_detections.extend(detections)
    
    tracker = Tracker([det.asdict() for det in all_detections])
    
    # visualization part
    vis_engine = VisEngine(vis_cfg, save_dir, tracker)
    for data in tqdm(dataset, desc='Visualization'):
        vis_engine.process(data)
    
def main():
    args = parse_args()
    track(**vars(args))
    
    
if __name__ == '__main__':
    main()