import os
import yaml
import argparse
import random
import torch

from tqdm import tqdm

from pbtrack.datasets.posetrack import ImageFolder
from pbtrack.tracker.tracker import Tracker
from pbtrack.visualization.vis_engine import VisEngine
from pbtrack.wrappers.torchreid2detections import Torchreid2detections  # need to import Torchreid2detections before
# StrongSORT2detections, so that 'bpbreid' is added to system path first
from pbtrack.wrappers.dekr2detections import DEKR2detections
from pbtrack.wrappers.strong_sort2detections import StrongSORT2detections


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str,
                        help='path to directory containing images')
    parser.add_argument('--vis-cfg', type=str,
                        default='configs/pbtrack/track.yaml',
                        help='path to visualization config file')
    parser.add_argument('--dekr-cfg', type=str,
                        default='configs/modules/detect/dekr/dekr.yaml',
                        help='path to dekr config file')
    parser.add_argument('--strongsort-cfg', type=str,
                        default='configs/modules/track/strongsort/track.yaml',
                        help='path to strongsort config file')
    parser.add_argument('--bpbreid-cfg', type=str, default='configs/modules/reid/bpbreid/local_bpbreid_train.yaml')
    parser.add_argument('--job-id', type=int,
                        help='Slurm job id', default=None)
    args = parser.parse_args()
    return args


def track(
    input_dir,
    vis_cfg,
    dekr_cfg,
    strongsort_cfg,
    bpbreid_cfg,
    job_id=random.randint(0, 1_000_000_000),
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

    # load reid
    model_reid = Torchreid2detections(
        device,
        save_dir,
        bpbreid_cfg,
        model_pose,
        job_id
    )

    # load model tracker
    # TODO make it modulable and more parametrable
    model_track = StrongSORT2detections(
        strongsort_cfg,
        device
    )
    
    # load dataset
    dataset = ImageFolder(input_dir)

    # train
    model_reid.train()

    # process images
    all_detections = []
    for data in tqdm(dataset, desc='Inference'): # tensor RGB (3, H, W)
        # pose estimation part -> create detections object
        detections, _ = model_pose.run(data)

        # reid part -> update detections object
        detections = model_reid.run(detections, data)

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