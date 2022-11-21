import os
import os.path as osp
import yaml
import argparse
from tqdm import tqdm

import torch

# FIXME ugly
import sys
sys.path.insert(0, './PoseTrack21/eval/posetrack21')
import PoseTrack21.eval.posetrack21.posetrack21.trackeval as trackeval

from lib.tracker import Tracker
from lib.datasets import PoseTrack
from lib.dekr2detections import DEKR2detections
from lib.strong_sort2detections import StrongSORT2detections


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-cfg', type=str,
                        default='configs/val.yaml',
                        help='path to evaluation config file')
    parser.add_argument('--dekr-cfg', type=str,
                        default='configs/dekr.yaml', 
                        help='path to dekr config file')
    parser.add_argument('--strongsort-cfg', type=str,
                        default='trackers/strong_sort/configs/track.yaml',
                        help='path to strongsort config file')
    args = parser.parse_args()
    return args

@torch.no_grad()
def val(
    eval_cfg,
    dekr_cfg,
    strongsort_cfg,
    *args, **kwargs
):
    # handle eval_cfg and paths
    with open(eval_cfg, 'r') as f:
        eval_cfg = yaml.load(f, Loader=yaml.FullLoader)
    save_dir = osp.join('runs', eval_cfg['save_dir'])
    if osp.exists(save_dir):
        i = 0
        while osp.exists(save_dir + str(i)): i += 1
        save_dir = save_dir + str(i)
        print(f"Save directory ({osp.join('runs', eval_cfg['save_dir'])}) already exists.")
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
    
    # load dataset
    dataset = PoseTrack(eval_cfg['posetrack_dir'], eval_cfg['subset'])
    
    # process images
    all_detections = []
    # FIXME add a tqdm by video (epoch) and a tqdm by frame (batch iteration)
    for data in tqdm(dataset, desc='Inference'): # tensor RGB (3, H, W)
        if data['frame'] == 1: # new video
            # TODO make it modulable and more parametrable
            # FIXME suboptimal to recreate a model at each new video
            model_track = StrongSORT2detections(
                strongsort_cfg,
                device
            )
        
        # pose estimation part -> create detections object
        detections = model_pose.run(data)
        
        # tracking part -> update detections object
        detections = model_track.run(data, detections)
        all_detections.extend(detections)
        
    tracker = Tracker([det.asdict() for det in all_detections])
    
    # PoseTrack21 evaluation scripts
    # FIXME clean a bit
    if eval_cfg['MOT']['eval']:
        mot_dir = osp.join(save_dir, 'mot')
        os.makedirs(mot_dir, exist_ok=True)
        tracker.save_mot(mot_dir)
        
        eval_cfg['MOT']['EVAL_CFG']['LOG_ON_ERROR'] = osp.join(mot_dir, 'error_log.txt')
        eval_cfg['MOT']['DATASET_CFG']['GT_FOLDER'] = osp.join(eval_cfg['posetrack_dir'], 
                                                               'posetrack_mot', 'mot', eval_cfg['subset'])
        eval_cfg['MOT']['DATASET_CFG']['TRACKERS_FOLDER'] = save_dir # ! not mot_dir
        evaluator = trackeval.EvaluatorMOT(eval_cfg['MOT']['EVAL_CFG'])
        dataset_list = [trackeval.datasets.PoseTrackMOT(eval_cfg['MOT']['DATASET_CFG'])]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA]:
            if metric.get_name() in eval_cfg['MOT']['METRICS_CFG']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)
            
    if eval_cfg['pose_estimation']['eval']:
        pose_estimation_dir = osp.join(save_dir, 'pose_estimation')
        os.makedirs(pose_estimation_dir, exist_ok=True)
        tracker.save_pose_estimation(pose_estimation_dir)
        
        eval_cfg['pose_estimation']['EVAL_CFG']['LOG_ON_ERROR'] = osp.join(pose_estimation_dir, 
                                                                           'error_log.txt')
        eval_cfg['pose_estimation']['DATASET_CFG']['GT_FOLDER'] = osp.join(eval_cfg['posetrack_dir'], 
                                                                           'posetrack_data', eval_cfg['subset'])
        eval_cfg['pose_estimation']['DATASET_CFG']['TRACKERS_FOLDER'] = pose_estimation_dir
        evaluator = trackeval.PoseEvaluator(eval_cfg['pose_estimation']['EVAL_CFG'])
        dataset_list = [trackeval.datasets.PoseTrack(eval_cfg['pose_estimation']['DATASET_CFG'])]
        metrics_list = []
        for metric in [trackeval.metrics.PosemAP]:
            if metric.get_name() in eval_cfg['pose_estimation']['METRICS_CFG']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)
    
    if any((eval_cfg['pose_tracking']['eval'], eval_cfg['reid_pose_tracking']['eval'])):
        pose_tracking_dir = osp.join(save_dir, 'pose_tracking')
        os.makedirs(pose_tracking_dir, exist_ok=True)
        tracker.save_pose_tracking(pose_tracking_dir)
    
    if eval_cfg['pose_tracking']['eval']:
        eval_cfg['pose_tracking']['EVAL_CFG']['LOG_ON_ERROR'] = osp.join(pose_tracking_dir, 
                                                                         'error_log_track.txt')
        eval_cfg['pose_tracking']['DATASET_CFG']['GT_FOLDER'] = osp.join(eval_cfg['posetrack_dir'], 
                                                                         'posetrack_data', eval_cfg['subset'])
        eval_cfg['pose_tracking']['DATASET_CFG']['TRACKERS_FOLDER'] = pose_tracking_dir
        evaluator = trackeval.Evaluator(eval_cfg['pose_tracking']['EVAL_CFG'])
        dataset_list = [trackeval.datasets.PoseTrack(eval_cfg['pose_tracking']['DATASET_CFG'])]
        metrics_list = []
        for metric in [trackeval.metrics.HOTAeypoints]:
            if metric.get_name() in eval_cfg['pose_tracking']['METRICS_CFG']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)
    
    # FIXME does not work
    if eval_cfg['reid_pose_tracking']['eval']:
        eval_cfg['reid_pose_tracking']['EVAL_CFG']['LOG_ON_ERROR'] = osp.join(pose_tracking_dir, 
                                                                              'error_log_reid_track.txt')
        eval_cfg['reid_pose_tracking']['DATASET_CFG']['GT_FOLDER'] = osp.join(eval_cfg['posetrack_dir'], 
                                                                              'posetrack_data', eval_cfg['subset'])
        eval_cfg['reid_pose_tracking']['DATASET_CFG']['TRACKERS_FOLDER'] = pose_tracking_dir
        evaluator = trackeval.EvaluatorReid(eval_cfg['reid_pose_tracking']['EVAL_CFG'])
        dataset_list = [trackeval.datasets.PoseTrackReID(eval_cfg['reid_pose_tracking']['DATASET_CFG'])]
        metrics_list = []
        for metric in [trackeval.metrics.HOTAReidKeypoints]:
            if metric.get_name() in eval_cfg['reid_pose_tracking']['METRICS_CFG']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)

def main():
    args = parse_args()
    val(**vars(args))
   
if __name__ == '__main__':
    main()