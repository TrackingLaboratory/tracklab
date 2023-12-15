#!/bin/bash

GT_FOLDER="${1:-/home/group-cvg/doering/2022/PoseTrackReIDDataset/release/PoseTrack21/posetrack_data/val/}"
TRACKERS_FOLDER="${2:-/home/group-cvg/doering/2022/PoseTrackReIDEvaluationData/example_pose_tracking_result_folder/}"

python3 scripts/run_posetrack_challenge.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --USE_PARALLEL True \
       --NUM_PARALLEL_CORES 8 
